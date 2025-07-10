#include <import>

int main(int argc, symbol argv[]) {
    map    args   = A_args(argv, "server", string("www.silver-lang.org"), null);
    string server = get(args, string("server"));

    // receive index for whom to 
    print("connecting to %o", args);
    
    // blocking api until code is stable, then translate to a-sync.
    // developing in async first is slower
    return 0;
}