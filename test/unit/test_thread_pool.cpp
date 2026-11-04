// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "libmlvc/common/thread_pool.hpp"

#include <libmlvc/error_codes.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace libmlvc;
using Priority = ThreadPool::Priority;
using Status = ThreadPool::Task::Status;

static_assert(!std::is_constructible_v<ThreadPool::Task, std::string, Priority>);

namespace {

class Event {
public:
    void Set()
    {
        {
            std::lock_guard lock(m_mutex);
            m_set = true;
        }
        m_cv.notify_all();
    }

    bool Wait()
    {
        std::unique_lock lock(m_mutex);
        return m_cv.wait_for(lock, std::chrono::seconds(10), [this] { return m_set; });
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_set{ false };
};

// Sets the flag when destroyed, after a delay so that checking it too early fails reliably
class ReleaseProbe {
public:
    explicit ReleaseProbe(std::shared_ptr<std::atomic<bool>> released) : m_released(std::move(released)) {}
    ReleaseProbe(const ReleaseProbe&) = delete;
    ReleaseProbe& operator=(const ReleaseProbe&) = delete;

    ~ReleaseProbe()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        *m_released = true;
    }

private:
    std::shared_ptr<std::atomic<bool>> m_released;
};

expected<void> Succeed()
{
    return {};
}

expected<void> Fail()
{
    return make_error_code(Error::general_failure);
}

// Signals `started`, then occupies its worker until `released` is set
std::function<expected<void>()> Block(Event& started, Event& released)
{
    return [&started, &released]() -> expected<void> {
        started.Set();
        released.Wait();
        return {};
    };
}

std::function<expected<void>()> Record(std::vector<int>& order, const int id)
{
    return [&order, id]() -> expected<void> {
        order.push_back(id);
        return {};
    };
}

}  // namespace

TEST(UnitTestThreadPool, CreateRejectsInvalidThreadCount)
{
    const auto pool = ThreadPool::Create(-1);
    ASSERT_FALSE(pool);
    EXPECT_EQ(pool.error(), make_error_code(Error::invalid_argument));
}

TEST(UnitTestThreadPool, RunsTasksInSubmitWithZeroThreads)
{
    std::thread::id threadId;
    auto pool = ThreadPool::Create(0);
    ASSERT_TRUE(pool) << pool.error().message();

    const auto task = pool.value()->Submit("task", [&]() -> expected<void> {
        threadId = std::this_thread::get_id();
        return {};
    });
    const auto failing = pool.value()->Submit("failing", Fail);

    EXPECT_EQ(task->GetStatus(), Status::COMPLETED);
    EXPECT_EQ(threadId, std::this_thread::get_id());
    EXPECT_EQ(failing->GetStatus(), Status::FAILED);
    EXPECT_EQ(failing->GetError(), make_error_code(Error::general_failure));
}

TEST(UnitTestThreadPool, RunsHigherPriorityFirstThenInSubmissionOrder)
{
    std::vector<int> order;
    Event started;
    Event released;
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    pool.value()->Submit("blocking", Block(started, released));
    ASSERT_TRUE(started.Wait());
    const std::pair<int, Priority> tasks[] = {
        { 0, Priority::LOW }, { 1, Priority::NORMAL }, { 2, Priority::HIGH },
        { 3, Priority::LOW }, { 4, Priority::HIGH },   { 5, Priority::NORMAL },
    };
    for (const auto& [id, priority] : tasks) {
        pool.value()->Submit("record", Record(order, id), priority);
    }
    pool.value()->Submit("record", Record(order, 6));  // Default priority (NORMAL)
    const auto last = pool.value()->Submit("last", Succeed, Priority::LOW);
    released.Set();

    ASSERT_EQ(last->Wait(), Status::COMPLETED);
    EXPECT_EQ(order, (std::vector{ 2, 4, 1, 5, 6, 0, 3 }));
}

TEST(UnitTestThreadPool, SetPriorityReordersQueuedTask)
{
    std::vector<int> order;
    Event started;
    Event released;
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    pool.value()->Submit("blocking", Block(started, released));
    ASSERT_TRUE(started.Wait());
    std::vector<std::shared_ptr<ThreadPool::Task>> tasks;
    for (int i = 0; i < 3; i++) {
        tasks.push_back(pool.value()->Submit("record", Record(order, i)));
    }
    const auto last = pool.value()->Submit("last", Succeed, Priority::LOW);
    tasks[2]->SetPriority(Priority::HIGH);
    released.Set();

    ASSERT_EQ(last->Wait(), Status::COMPLETED);
    EXPECT_EQ(order, (std::vector{ 2, 0, 1 }));
}

TEST(UnitTestThreadPool, ConcurrentSetPriorityKeepsOtherTasksInOrder)
{
    // Task 0 flips between HIGH and LOW while the worker picks the next task; 1 (HIGH) must still run before 2 (NORMAL)
    for (int i = 0; i < 200; i++) {
        std::vector<int> order;
        Event started;
        Event released;
        auto pool = ThreadPool::Create();
        ASSERT_TRUE(pool) << pool.error().message();

        pool.value()->Submit("blocking", Block(started, released));
        ASSERT_TRUE(started.Wait());
        const auto flipping = pool.value()->Submit("record", Record(order, 0), Priority::HIGH);
        const auto high = pool.value()->Submit("record", Record(order, 1), Priority::HIGH);
        const auto normal = pool.value()->Submit("record", Record(order, 2), Priority::NORMAL);
        released.Set();
        while (normal->GetStatus() == Status::QUEUED) {
            flipping->SetPriority(Priority::LOW);
            flipping->SetPriority(Priority::HIGH);
        }
        flipping->Wait();
        high->Wait();
        normal->Wait();

        ASSERT_TRUE(std::ranges::find(order, 1) < std::ranges::find(order, 2)) << "iteration " << i;
    }
}

TEST(UnitTestThreadPool, ReportsTaskState)
{
    Event started;
    Event released;
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    const auto running = pool.value()->Submit("running", Block(started, released));
    ASSERT_TRUE(started.Wait());
    const auto queued = pool.value()->Submit("queued", Succeed);

    EXPECT_EQ(running->GetName(), "running");
    EXPECT_EQ(queued->GetName(), "queued");
    EXPECT_EQ(running->GetStatus(), Status::RUNNING);
    EXPECT_EQ(queued->GetStatus(), Status::QUEUED);
    released.Set();

    ASSERT_EQ(queued->Wait(), Status::COMPLETED);
    EXPECT_EQ(running->GetStatus(), Status::COMPLETED);
}

TEST(UnitTestThreadPool, ReportsTaskFailure)
{
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    const auto failing = pool.value()->Submit("failing", Fail);
    const auto succeeding = pool.value()->Submit("succeeding", Succeed);

    EXPECT_EQ(failing->Wait(), Status::FAILED);
    EXPECT_EQ(failing->GetError(), make_error_code(Error::general_failure));
    EXPECT_EQ(succeeding->Wait(), Status::COMPLETED);
    EXPECT_FALSE(succeeding->GetError());
}

TEST(UnitTestThreadPool, WaitsForTaskCompletionFromMultipleThreads)
{
    std::atomic<bool> finished{ false };
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    const auto task = pool.value()->Submit("task", [&]() -> expected<void> {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        finished = true;
        return {};
    });
    std::atomic<Status> otherStatus{ Status::QUEUED };
    std::thread other([&] { otherStatus = task->Wait(); });
    EXPECT_EQ(task->Wait(), Status::COMPLETED);
    other.join();

    EXPECT_TRUE(finished);
    EXPECT_EQ(otherStatus.load(), Status::COMPLETED);
}

TEST(UnitTestThreadPool, RunsTasksSubmittedFromTasks)
{
    std::shared_ptr<ThreadPool::Task> inner;
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    auto* const poolPtr = pool.value().get();
    const auto outer = poolPtr->Submit("outer", [&]() -> expected<void> {
        inner = poolPtr->Submit("inner", Succeed);
        return {};
    });
    ASSERT_EQ(outer->Wait(), Status::COMPLETED);
    EXPECT_EQ(inner->Wait(), Status::COMPLETED);
}

TEST(UnitTestThreadPool, RunsTasksInParallel)
{
    Event secondStarted;
    auto pool = ThreadPool::Create(2);
    ASSERT_TRUE(pool) << pool.error().message();

    const auto first = pool.value()->Submit("first", [&]() -> expected<void> {
        if (!secondStarted.Wait()) return make_error_code(Error::general_failure);
        return {};
    });
    pool.value()->Submit("second", [&]() -> expected<void> {
        secondStarted.Set();
        return {};
    });
    EXPECT_EQ(first->Wait(), Status::COMPLETED);
}

TEST(UnitTestThreadPool, RunsEachTaskExactlyOnce)
{
    constexpr int numProducers = 4;
    constexpr int numTasksPerProducer = 250;
    constexpr int numTasks = numProducers * numTasksPerProducer;
    std::vector<std::atomic<int>> numRuns(numTasks);
    std::atomic<int> numRemaining{ numTasks };
    Event done;
    auto pool = ThreadPool::Create(2);
    ASSERT_TRUE(pool) << pool.error().message();

    std::vector<std::thread> producers;
    for (int producer = 0; producer < numProducers; producer++) {
        producers.emplace_back([&, producer] {
            for (int i = 0; i < numTasksPerProducer; i++) {
                pool.value()->Submit("task", [&, id = producer * numTasksPerProducer + i]() -> expected<void> {
                    numRuns[id]++;
                    if (--numRemaining == 0) done.Set();
                    return {};
                });
            }
        });
    }
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_TRUE(done.Wait());
    EXPECT_EQ(std::ranges::count_if(numRuns, [](const auto& runs) { return runs.load() == 1; }), numTasks);
}

TEST(UnitTestThreadPool, ReleasesWorkBeforeReportingResult)
{
    const auto released = std::make_shared<std::atomic<bool>>(false);
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    auto probe = std::make_shared<ReleaseProbe>(released);
    const auto task = pool.value()->Submit("task", [probe = std::move(probe)]() -> expected<void> { return {}; });

    ASSERT_EQ(task->Wait(), Status::COMPLETED);
    EXPECT_TRUE(*released);
}

TEST(UnitTestThreadPool, DestructorCancelsQueuedTasksBeforeWaitingForRunningTask)
{
    // The running task waits for a queued one, which only the destructor can finish by cancelling it
    std::shared_ptr<ThreadPool::Task> queued;
    std::shared_ptr<ThreadPool::Task> late;
    std::atomic<Status> queuedStatus{ Status::QUEUED };
    std::atomic<bool> releasedBeforeCancelled{ false };
    const auto released = std::make_shared<std::atomic<bool>>(false);
    Event started;
    Event submitted;
    auto pool = ThreadPool::Create();
    ASSERT_TRUE(pool) << pool.error().message();

    auto* const poolPtr = pool.value().get();
    const auto running = poolPtr->Submit("running", [&]() -> expected<void> {
        started.Set();
        if (!submitted.Wait()) return make_error_code(Error::general_failure);
        queuedStatus = queued->Wait();
        releasedBeforeCancelled = released->load();
        late = poolPtr->Submit("late", Succeed);
        return {};
    });
    ASSERT_TRUE(started.Wait());
    auto probe = std::make_shared<ReleaseProbe>(released);
    queued = poolPtr->Submit("queued", [probe = std::move(probe)]() -> expected<void> { return {}; });
    submitted.Set();
    pool.value().reset();

    EXPECT_EQ(running->GetStatus(), Status::COMPLETED);
    EXPECT_EQ(queuedStatus.load(), Status::CANCELLED);
    EXPECT_TRUE(releasedBeforeCancelled);
    ASSERT_TRUE(late);
    EXPECT_EQ(late->GetStatus(), Status::CANCELLED);
}
