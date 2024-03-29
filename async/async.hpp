#pragma once

#include <mx/mx.hpp>
#include <condition_variable>
#include <future>

namespace ion {

/// executables are different types, and require async
struct exec:str {
    exec(str s) : str(s) { }
    ion::path path() {
        return data;
    }
};

/// 
struct completer:mx {
    struct cdata {
        mx              vdata;
        mutex           mtx;
        bool            completed;
        array<mx>       l_success;
        array<mx>       l_failure;
    };
    mx_object(completer, mx, cdata);

    completer(lambda<void(mx)>& fn_success,
              lambda<void(mx)>& fn_failure) : completer() {
        
        fn_success = [data=data](mx d) { /// should not need a grab and drop; if
            data->completed = true;
            for (mx &s: data->l_success) {
                FnFuture &lambda = s.mem->ref<FnFuture>();
                lambda(d);
            }
        };
        
        fn_failure = [data=data](mx d) {
            data->completed = true;
            for (mx &f: data->l_failure) {
                FnFuture &lambda = f.mem->ref<FnFuture>();
                lambda(d);
            }
        };
    }
};

/// how do you create a future?  you give it completer memory
struct future:mx {
    completer::cdata &cd;
    future (memory*           mem) : mx(mem),         cd(ref<completer::cdata>()) { }
    future (const completer &comp) : mx(comp.hold()), cd(ref<completer::cdata>()) { }
    
    int sync() {
        cd.mtx.lock();
        mutex mtx;
        if (!cd.completed) {
            mtx.lock();
            void *v_mtx = &mtx;
            FnFuture fn = [v_mtx] (mx) { ((mutex*)v_mtx)->unlock(); };
            mx mx_fn = fn.mem->hold();
            cd.l_success.push(mx_fn);
            cd.l_failure.push(mx_fn);
        }
        cd.mtx.unlock();
        mtx.lock();
        return 0;
    };

    ///
    future& then(FnFuture fn) {
        cd.mtx.lock();
        cd.l_success += fn.mem->hold();
        cd.mtx.unlock();
        return *this;
    }

    ///
    future& except(FnFuture fn) {
        cd.mtx.lock();
        cd.l_failure += fn.mem->hold();
        cd.mtx.unlock();
        return *this;
    }
};

struct runtime;
typedef lambda<mx(runtime*, int)> FnProcess;

typedef std::condition_variable ConditionV;
struct runtime {
    memory                    *handle;   /// handle on rt memory
    mutex                      mtx_self; /// output the memory address of this mtx.
    lambda<void(mx)>           on_done;  /// not the same prototype as FnFuture, just as a slight distinguisher we dont need to do a needless non-conversion copy
    lambda<void(mx)>           on_fail;
    size_t                     count;
    FnProcess				   fn;
    array<mx>                  results;
    std::vector<std::thread>  *threads;       /// todo: unstd when it lifts off ground. experiment complete time to crash and blast in favor of new matrix.
    int                        done      = 0; /// 
    bool                       failure   = false;
    bool                       join      = false;
    bool                       stop      = false; /// up to the user to implement this effectively.  any given service isnt going to stop without first checking
};

struct process:mx {
protected:
    inline static bool init;
    
public:
    inline static ConditionV       cv_cleanup;
    inline static std::thread      th_manager;
    inline static mutex            mtx_global;
    inline static mutex            mtx_list;
    inline static doubly<runtime*> procs;
    inline static int              exit_code = 0;

    ///
    mx_object(process, mx, runtime);

    static inline void manager() {
        std::unique_lock<mutex> lock(mtx_list);
        for (bool quit = false; !quit;) {
            cv_cleanup.wait(lock);
            bool cycle    = false;
            do {
                cycle     = false;
                num index = 0;
                ///
                for (runtime *state: procs) {
                    state->mtx_self.lock();
                    auto &ps = *state;
                    if (ps.done == ps.threads->size()) {
                        ps.mtx_self.unlock();
                        lock.unlock();

                        /// join threads
                        for (auto &t: *(ps.threads))
                            t.join();
                        
                        /// manage process
                        ps.mtx_self.lock();
                        ps.join = true;

                        /// 
                        procs->remove(index); /// remove -1 should return the one on the end, if it exists; this is a bool result not some integer of index to treat as.
                       (procs->len() == 0) ?
                            (quit = true) : (cycle = true);
                        lock.lock();
                        ps.mtx_self.unlock();
                        break;
                    }
                    index++;
                    ps.mtx_self.unlock();
                }
            } while (cycle);
            /// dont unlock here because of the implicit behaviour of condition_variable
        }
        lock.unlock();
    }

    static void run(runtime *rt, int w) {
        runtime *data = rt;
        /// run (fn) the work (p) on this thread (i)
        mx r = data->fn(data, w);
        data->mtx_self.lock();

        data->failure |= !r;
        data->results[w] = r;
        
        /// wait for completion of one (we coudl combine c check inside but not willing to stress test that atm)
        mtx_global.lock();
        bool im_last = (++data->done >= data->count);
        mtx_global.unlock();

        /// if all complete, notify condition var after calling completer/failure methods
        if (im_last) {
            if (data->on_done) {
                if (data->failure)
                    data->on_fail(data->results);
                else
                    data->on_done(data->results);
            }
            mtx_global.lock();
            cv_cleanup.notify_all();
            mtx_global.unlock();
        }
        data->mtx_self.unlock();

        /// wait for job set to complete
        for (; data->done != data->count;)
            yield();
    }
    ///
    process(size_t count, FnProcess fn) : process(alloc<process>()) {
        if(!init) {
            init       = true;
            th_manager = std::thread(manager);
        }
        data->fn = fn;
        if (count) {
            data->count   = count;
            data->results = array<mx>(count);
        }
    }
    inline bool joining() const { return data->join; }
};

/// async is out of sync with the other objects.
struct async {
    ///
    struct delegation {
        process         proc;
        mx              results;
        mutex           mtx; /// could be copy ctr
    } d;

    async();
    
    /// n processes
    async(size_t, FnProcess);

    /// path to execute, we dont host a bunch of environment vars. ar.
    async(exec);
    
    /// singleton process
    async(FnProcess);

    async& operator=(const async &c);

    array<mx> sync(bool force_stop = false);

    /// await all async processes to complete
    static int await_all();

    /// return future for this async
    operator future();
};

/// sync just performs sync on construction
struct sync:async {
    sync(size_t count, FnProcess fn) : async(count, fn) {
        async::sync();
    }
    sync(exec p) : async(p) {
        async::sync();
    }

    /// call array<S> src -> T conversion
    template <typename T>
    operator array<T>() { return async::sync(); }
    operator      int() { return async::sync(); }
};
}
