// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/thread_pool.hpp"
#include "libmlvc/common/logging.hpp"
#include "libmlvc/common/macros.hpp"
#include "libmlvc/common/utils.hpp"

#include <libmlvc/error_codes.hpp>

#include <iterator>
#include <utility>

namespace libmlvc {

ThreadPool::Task::Status ThreadPool::Task::Wait() const
{
    auto status = m_status.load();
    while (status == Status::QUEUED || status == Status::RUNNING) {
        m_status.wait(status);
        status = m_status.load();
    }
    return status;
}

ThreadPool::Task::Task(std::string name, const Priority priority) : m_name(std::move(name)), m_priority(priority) {}

void ThreadPool::Task::SetStatus(const Status status)
{
    m_status = status;
    m_status.notify_all();
}

void ThreadPool::Task::SetResult(const expected<void>& result)
{
    if (result) {
        SetStatus(Status::COMPLETED);
    } else {
        m_error = result.error();
        SetStatus(Status::FAILED);
    }
}

expected<std::unique_ptr<ThreadPool>> ThreadPool::Create(const int numThreads)
{
    if (numThreads < 0) {
        MLVC_LOG_ERROR("Invalid thread pool size: %d", numThreads);
        return make_error_code(Error::invalid_argument);
    }

    auto pool = std::unique_ptr<ThreadPool>(new ThreadPool());
    for (int i = 0; i < numThreads; i++) {
        pool->m_threads.emplace_back(&ThreadPool::RunTasks, pool.get());
    }
    return pool;
}

ThreadPool::~ThreadPool()
{
    std::deque<WorkItem> queued;
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
        queued.swap(m_queue);
    }
    m_cv.notify_all();
    // Cancel before joining, as running tasks may be waiting for queued ones
    for (auto& item : queued) {
        CancelTask(item);
    }
    for (auto& thread : m_threads) {
        thread.join();
    }
}

std::shared_ptr<ThreadPool::Task> ThreadPool::Submit(std::string name, std::function<expected<void>()> work,
                                                     const Priority priority)
{
    MLVC_ASSERT(work);
    auto task = std::shared_ptr<Task>(new Task(std::move(name), priority));
    WorkItem item{ .task = task, .work = std::move(work) };
    if (m_threads.empty()) {
        RunTask(item);
        return task;
    }

    std::unique_lock lock(m_mutex);
    if (m_stop) {
        // Submitted by a running task while the pool is being destroyed
        lock.unlock();
        CancelTask(item);
        return task;
    }
    m_queue.push_back(std::move(item));
    lock.unlock();
    m_cv.notify_one();
    return task;
}

void ThreadPool::RunTasks()
{
    while (true) {
        std::unique_lock lock(m_mutex);
        m_cv.wait(lock, [this] { return m_stop || !m_queue.empty(); });
        if (m_stop) {
            return;
        }
        // Read each priority once, as SetPriority can change them during the scan
        auto next = m_queue.begin();
        auto nextPriority = next->task->m_priority.load();
        for (auto it = std::next(next); it != m_queue.end(); ++it) {
            if (const auto priority = it->task->m_priority.load(); priority < nextPriority) {
                next = it;
                nextPriority = priority;
            }
        }
        auto item = std::move(*next);
        m_queue.erase(next);
        lock.unlock();

        RunTask(item);
    }
}

void ThreadPool::RunTask(WorkItem& item)
{
    const auto& name = item.task->GetName();
    item.task->SetStatus(Task::Status::RUNNING);
    MLVC_LOG_INFO("[%s] Running task...", name.c_str());
    ManualOpTimer timer;
    const auto ret = item.work();
    const auto durationMs = timer.Stop();
    if (ret) {
        MLVC_LOG_INFO("[%s] Task completed in %.2f ms", name.c_str(), durationMs);
    } else {
        MLVC_LOG_ERROR("[%s] Task failed in %.2f ms: %s", name.c_str(), durationMs, ret.error().message().c_str());
    }
    // Release the captures before waiters can observe the result
    item.work = nullptr;
    item.task->SetResult(ret);
}

void ThreadPool::CancelTask(WorkItem& item)
{
    MLVC_LOG_INFO("[%s] Task cancelled", item.task->GetName().c_str());
    item.work = nullptr;
    item.task->SetStatus(Task::Status::CANCELLED);
}

}  // namespace libmlvc
