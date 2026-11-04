// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <libmlvc/expected.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace libmlvc {

class ThreadPool {
public:
    enum struct Priority { HIGH = 0, NORMAL = 1, LOW = 2 };

    class Task {
    public:
        enum struct Status { QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED };

        void SetPriority(const Priority priority) { m_priority = priority; }
        const std::string& GetName() const { return m_name; }
        Status GetStatus() const { return m_status; }
        std::error_code GetError() const { return m_status == Status::FAILED ? m_error : std::error_code{}; }
        Status Wait() const;

    private:
        friend class ThreadPool;

        std::string m_name;
        std::error_code m_error;
        std::atomic<Priority> m_priority;
        std::atomic<Status> m_status{ Status::QUEUED };

        Task(std::string name, Priority priority);
        void SetStatus(Status status);
        void SetResult(const expected<void>& result);
    };

    // With zero threads, Submit runs the work on the calling thread before returning.
    static expected<std::unique_ptr<ThreadPool>> Create(int numThreads = 1);

    // Cancels queued tasks, then waits for running tasks to finish. Must not run on a worker thread, which includes a
    // task releasing the last owner of the pool.
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // The work must report failures through its result instead of throwing. The work and its captures are released
    // before the task reports its final status. Work submitted while the pool is being destroyed is cancelled.
    std::shared_ptr<Task> Submit(std::string name, std::function<expected<void>()> work,
                                 Priority priority = Priority::NORMAL);

private:
    struct WorkItem {
        std::shared_ptr<Task> task;
        std::function<expected<void>()> work;
    };

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<WorkItem> m_queue;
    bool m_stop{ false };
    std::vector<std::thread> m_threads;

    ThreadPool() = default;
    void RunTasks();
    static void RunTask(WorkItem& item);
    static void CancelTask(WorkItem& item);
};

}  // namespace libmlvc
