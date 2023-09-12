#include <mx/mx.hpp>
#include <sys/stat.h>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

namespace ion {

logger console;

void free64(void* ptr) {
#if defined(WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void* calloc64(size_t count, size_t size) {
    // Calculate total bytes
    size_t total_bytes = count * size;

    // We'll use aligned_alloc on platforms that support it
#if defined(WIN32)
    void* ptr = _aligned_malloc(total_bytes, 64);
    if (!ptr) {
        return NULL;
    }
    memset(ptr, 0, total_bytes);
    return ptr;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    void* ptr = aligned_alloc(64, total_bytes);
    if (!ptr) {
        return NULL;
    }
    memset(ptr, 0, total_bytes);
    return ptr;
#else
    // For platforms that don't support C11's aligned_alloc
    void* ptr = NULL;
    // posix_memalign allocates aligned memory
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return NULL;  // posix_memalign will return nonzero on failure
    }
    memset(ptr, 0, total_bytes);
    return ptr;
#endif
}

raw_t memory::typed_data(type_t dtype, size_t index) const {
    size_t mxc = math::max(reserve, count);
    //static type_t mx_t  = typeof(mx);
    alloc_schema *schema = type->schema;
    if (dtype != type && schema) {
        size_t offset = 0;
        for (size_t i = 0; i < schema->bind_count; i++) {
            context_bind &c = schema->composition[i];
            if (c.data == dtype)
                return (raw_t)&cstr(origin)[c.offset * mxc + c.data->base_sz * index];
        }
        console.fault("type not found in schema: {0}", { str(dtype->name) });
        return (raw_t)null;
    } else
        return (raw_t)(cstr(origin) + dtype->base_sz * index);
}

size_t length(std::ifstream& in) {
    std::streamsize base = in.tellg();
    in.seekg(0, std::ios::end);
    std::streamsize to_end = in.tellg();
    in.seekg(base);
    return to_end - base;
}

size_t idata::size() {
    return schema ? schema->total_bytes : base_sz;
}

memory *idata::lookup(symbol sym) {
    u64 hash = djb2(cstr(sym));
    memory **result = symbols ? symbols->djb2.lookup(hash) : null;
    return   result ? *result : null;
}

memory *idata::lookup(i64 id) {
    memory **result = symbols ? symbols->ids.lookup(id) : null;
    return   result ? *result : null;
}

///
int str::index_of(MatchType ct, symbol mp) const {
    int  index = 0;
    
    using Fn = func<bool(const char &)>;
    static umap<MatchType, Fn> match_table {
        { Alpha,     Fn([&](const char &c) -> bool { return  isalpha (c); }) },
        { Numeric,   Fn([&](const char &c) -> bool { return  isdigit (c); }) },
        { WS,        Fn([&](const char &c) -> bool { return  isspace (c); }) }, // lambda must used copy-constructed syntax
        { Printable, Fn([&](const char &c) -> bool { return !isspace (c); }) },
        { String,    Fn([&](const char &c) -> bool { return  strcmp  (&c, mp) == 0; }) },
        { CIString,  Fn([&](const char &c) -> bool { return  strcmp  (&c, mp) == 0; }) }
    };
    
    /// msvc thinks its ambiguous, so i am removing this iterator from str atm.
    cstr pc = data;
    for (;;) {
        char &c = pc[index];
        if  (!c)
            break;
        if (match_table[ct](c))
            return index;
        index++;
    }
    return -1;
}

int snprintf(cstr str, size_t size, const char *format, ...) {
    int n;
    va_list args;
    va_start(args, format);
    
#ifdef _MSC_VER
    n = _vsnprintf_s(str, size, _TRUNCATE, format, args);
#else
    n = vsnprintf(str, size, format, args);
#endif
    
    va_end(args);
    if (n < 0 || n >= (int)size) {
        // handle error here
    }
    return n;
}

str path::mime_type() {
    mx e = ext().mid(1).symbolize();
    static path  data = "data/mime-type.json";
    static map<mx> js =  data.read<var>();
    field<mx> *find = js->lookup(e);
    return find ? find->value.grab() : ((memory*)js["default"])->grab();
}

i64 integer_value(memory *mem) {
    symbol   c = mdata<char>(mem, 0);
    bool blank = c[0] == 0;
    while (isalpha(*c))
        c++;
    return blank ? i64(0) : i64(strtol(c, nullptr, 10));
}

memory *drop(memory *mem) {
    if (mem) mem->drop();
    return mem;
}

memory *grab(memory *mem) {
    if (mem) mem->grab();
    return mem;
}

i64 millis() {
    return i64(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

/// attach arbs to memory (uses a pointer)
attachment *memory::attach(ion::symbol id, void *data, func<void()> release) {
    if (!atts)
         atts = new doubly<attachment>();
    doubly<attachment> &a = *atts;
    a->push(attachment {id, data, release});
    return &a->last();
}

attachment *memory::find_attachment(ion::symbol id) {
    if (!atts) return nullptr;
    /// const char * symbol should work fine for the cases used
    for (attachment &att:*atts)
        if (id == att.id)
            return &att;
    return nullptr;
}

void *memory::realloc(size_t alloc_reserve, bool fill_default) {
    size_t          type_sz   = type->size(); /// size of actual data consumed -- we dont use functions that are emitted from array<T> we use T in vector form (which they are already)
    u8             *dst       = (u8*)calloc64(alloc_reserve, type_sz);
    u8             *src       = (u8*)origin;
    size_t          mn        = math::min(alloc_reserve, count);
    const bool      prim      = (type->traits & traits::primitive) != 0;

    /// if single primitive, it can be mem copied.  otherwise this is interleaved vector
    if (prim) {
        memcpy(dst, src, type_sz * mn);
    } else {
        for (size_t i = 0; i < type->schema->bind_count; i++) {
            context_bind &c  = type->schema->composition[i];
            for (size_t ii = 0; ii < mn; ii++) {
                const bool data_prim = (c.data->traits & traits::primitive) != 0;
                if (data_prim) {
                    memcpy(&dst[c.offset + ii * type_sz], &src[c.offset + ii * type_sz], type_sz);
                } else {
                    c.data->functions->copy    (raw_t(0), &dst[c.offset + ii * type_sz], &src[c.offset + ii * type_sz]); /// copy prior data
                    c.data->functions->destruct(raw_t(0), &src[c.offset + ii * type_sz]); /// destruct prior data
                }
            }
        }
    }
    /// this controls the 'count' which is in-effect the constructed count.  if we are not constructing, no updating count
    if (fill_default) {
        count = alloc_reserve;
        if (!prim) {
            for (size_t i = 0; i < type->schema->bind_count; i++) {
                context_bind &c  = type->schema->composition[i];
                if (!(c.data->traits & traits::primitive))
                    for (size_t ii = mn; ii < alloc_reserve; ii++)
                        c.data->functions->construct(raw_t(0), &dst[c.offset + ii * type_sz]);
            }
        }
    }
    free64(origin);
    origin  = raw_t(dst);
    reserve = alloc_reserve;
    return origin;
}

/// put primitive conversions here, and mx.hpp needs the declarations
memory *_to_string(cstr data) {
    return memory::stringify(data, memory::autolen, 0, false);
}

size_t  _hash(cstr data, size_t count) {
    return djb2(data, count);
}

/// mx-objects are clearable which brings their count to 0 after destruction
void memory::clear() {
    alloc_schema *mx  = type->schema;
    u8           *dst = (u8*)origin;
    if (mx) {
        for (size_t i = 0; i < mx->bind_count; i++) { /// count should be called bind_count or something; its too ambiguous with memory
            context_bind &c  = mx->composition[i];
            for (size_t ii = 0; ii < count; ii++)
                c.data->functions->destruct(raw_t(0), &dst[c.offset + ii * mx->total_bytes]);
        }
    }
    count = 0;
}

memory *memory::stringify(cstr cs, size_t len, size_t rsv, bool constant, type_t ctype, i64 id) {
    ion::symbol sym = (ion::symbol)(cs ? cs : "");
    if (constant) {
        if(!ctype->symbols)
            ctype->symbols = new symbol_data { };
        u64  h_sym = djb2(cstr(sym));
        memory *&m = ctype->symbols->djb2[h_sym];
        if (!m) {
            size_t ln = (len == memory::autolen) ? strlen(sym) : len; /// like auto-wash
            m = memory::alloc(typeof(char), ln, ln + 1, raw_t(sym));
            m->id    = id;
            m->attrs = attr::constant;
            ctype->symbols->ids[id] = m; /// was not hashing by id, it was the djb2 again (incorrect)
            //ctype->symbol_djb2[h_sym] = m; 'redundant due to the setting of the memory*& (which [] operator always inserts item)
            ctype->symbols->list->push(m);
        }
        return m->grab();
    } else {
        size_t     ln = (len == memory::autolen) ? strlen(sym) : len;
        size_t     al = (rsv >= (ln + 1)) ? rsv : (ln + 1);
        memory*   mem = memory::alloc(typeof(char), ln, al, raw_t(sym));
        mem->id       = id;
        cstr  start   = mem->data<char>(0);
        start[ln]     = 0;
        return mem;
    }
}

memory *memory::string (std::string s) { return stringify(cstr(s.c_str()), s.length(), 0, false, typeof(char), 0); }
memory *memory::cstring(cstr s)        { return stringify(cstr(s),         strlen(s),  0, false, typeof(char), 0); }

memory *memory::symbol (ion::symbol s, type_t ty, i64 id) {
    return stringify(cstr(s), strlen(s), 0, true, ty, id);
}

memory *memory::raw_alloc(type_t type, size_t sz, size_t count, size_t res) {
    size_t elements = math::max(count, res);
    memory*     mem = (memory*)calloc64(1, sizeof(memory)); /// there was a 16 multiplier prior.  todo: add address sanitizer support with appropriate clang stuff
    mem->count      = count;
    mem->reserve    = math::max(res, count);
    mem->refs       = 1;
    mem->type       = type;
    mem->origin     = sz ? calloc64(sz, mem->reserve) : null; /// was doing inline origin.  its useful prior to realloc but adds complexity; can add back when optimizing
    return mem;
}

#ifdef WIN32
void usleep(__int64 usec) {
    HANDLE timer; 
    LARGE_INTEGER ft; 

    ft.QuadPart = -(10*usec); // Convert to 100 nanosecond interval, negative value indicates relative time

    timer = CreateWaitableTimer(NULL, TRUE, NULL); 
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0); 
    WaitForSingleObject(timer, INFINITE); 
    CloseHandle(timer); 
}
#endif

/// starting at 1, it should remain active.  shall not be freed as a result
void memory::drop() {
    if (--refs <= 0 && !constant) { /// <= because mx_object does a defer on the actual construction of the container class
        origin = null;
        // delete attachment lambda after calling it
        if (atts) {
            for (attachment &att: *atts)
                att.release();
            delete atts;
            atts = null;
        }
        if (managed)
            free64(origin);
        if (shape) {
            delete shape;
            shape = null;
        }
        free64(this);
    }
}

bool path::get_modified_date(struct tm *res) {
    const char* filename = cs();
    struct stat file_stat;

    if (stat(filename, &file_stat) == -1) {
        perror("stat");
        return false;
    }
    static mutex mx;
    mx.lock();
    *res = *gmtime(&file_stat.st_mtime);
    mx.unlock();
    return true;
}

str path::get_modified_string() {
    str result(size_t(128));
    struct tm gm_time;
    ///
    if (get_modified_date(&gm_time)) {
        // RFC 1123 format (used with http)
        if (strftime(result.data, result.reserve(), "%a, %d %b %Y %H:%M:%S GMT", &gm_time) == 0)
            fprintf(stderr, "strftime failed\n");
    }
    return result;
}

memory *memory::alloc(type_t type, size_t count, size_t reserve, raw_t v_src) {
    size_t  type_sz = type->size(); /// this is the 'data size', should name the function just that; if the type has no schema the data size is its own size

    if (type->singleton)
        return type->singleton->grab();
    
    memory *mem = memory::raw_alloc(type, type_sz, count, reserve);

    if (type->traits & traits::singleton)
        type->singleton = mem;
    bool primitive = (type->traits & traits::primitive);

    /// if allocating a schema-based object (mx being first user of this)
    if (count > 0) {
        if (v_src) {
            if (primitive)
                memcpy(mem->origin, v_src, type_sz * count);
            else {
                /// if schema-copy-construct (call cpctr for each data type in composition)
                for (size_t i = 0; i < type->schema->bind_count; i++) {
                    context_bind &bind = type->schema->composition[i];
                    u8 *dst = &((u8*)mem->origin)[bind.offset];
                    u8 *src = &((u8*)      v_src)[bind.offset];
                    if (bind.data) {
                        for (size_t ii = 0; ii < count; ii++)
                            if (bind.data->traits & traits::primitive)
                                memcpy(&dst[ii * type_sz], &src[ii * type_sz], type_sz);
                            else
                                bind.data->functions->copy(raw_t(0), &dst[ii * type_sz], &src[ii * type_sz]);
                    }
                }
            }
        } else if (!primitive) {
            if (!type->schema) {
                /// restrict to structs with no initialization needed and simple bit copies suffice
                size_t type_sz = type->base_sz;
                u8 *dst = &((u8*)mem->origin)[0];
                if (v_src) {
                    u8 *src = &((u8*)v_src)[0];
                    memcpy(dst, src, count * type_sz);
                } else {
                    memset(dst, 0, count * type_sz);
                }
            } else {
                for (size_t i = 0; i < type->schema->bind_count; i++) {
                    context_bind &bind = type->schema->composition[i];
                    u8 *dst  = &((u8*)mem->origin)[bind.offset];
                    if (bind.data && !(bind.data->traits & traits::primitive))
                        for (size_t ii = 0; ii < count; ii++) {
                            bind.data->functions->construct(raw_t(0), &dst[ii * type_sz]);
                        }
                }
            }

        }
    }
    return mem;
}

memory *memory::copy(size_t reserve) {
    memory *a   = this;
    memory *res = alloc(a->type, a->count, reserve, a->origin);
    return  res;
}

memory *memory::grab() {
    refs++;
    return this;
}

size &size::operator=(const size b) {
    memcpy(values, b.values, sizeof(values));
    count = b.count;
    return *this;
}


void chdir(std::string c) {
#if defined(_WIN32)
    // replace w global for both cases; its not worth it
    //SetCurrentDirectory(c.c_str());
#else
    ::chdir(c.c_str());
#endif
}

memory* mem_symbol(ion::symbol cs, type_t ty, i64 id) {
    return memory::symbol(cs, ty, id);
}

void *mem_origin(memory *mem) {
    return mem->origin;
}

memory *cstring(cstr cs, size_t len, size_t reserve, bool is_constant) {
    return memory::stringify(cs, len, 0, is_constant, typeof(char), 0);
}

idata *types::lookup(str &name) {
    memory *sym = name.symbolize();
    type_t type = (*types::type_map)[(symbol)sym->origin];
    return type;
}

void types::hash_type(type_t type) {
    str  type_name = type->name;
    memory    *sym = type_name.symbolize(); // issue with using symbol table on type bootstrap; the hmap and doubly would have to set flags to avoid the schema registration
    if (!types::type_map)
         types::type_map = new hmap<ion::symbol, type_t>(64);
    type_t &n_data = (*types::type_map)[(symbol)sym->origin];
    n_data = type;
}

u8* get_member_address(type_t type, raw_t data, str &name, prop *&rprop) {
    if (type->meta && type->meta_map) {
        memory *sym = name.symbolize();
        prop  **p   = type->meta_map->lookup((symbol)sym->origin);
        if (p) {
            rprop       = *p;
            u8 *p_value = &(((u8*)data)[(*p)->offset]);
            return p_value;
        }
    }
    return null;
}

/// find schema-bound meta property from memory and field
u8* property_find(void *origin, type_t type, str &name, prop *&rprop) {
    u8  *pos = (u8*)origin;
    type_t t =      type;

    if (t->schema)
        for (int i = 0; i < t->schema->bind_count; i++) {
            context_bind &c = t->schema->composition[i];
            prop*    member = null;
            u8*      addr   = get_member_address(c.data, pos, name, member);
            if (addr) {
                rprop = member;
                return addr;
            }
            pos += c.data->base_sz;
        }
    rprop = null;
    return null;
}


bool get_bool(type_t type, raw_t data, str &name) {
    prop  *p;
    u8    *p_value = get_member_address(type, data, name, p);
    bool   result  = p->member_type->functions->boolean(null, p_value);
    return result;
}

memory *get_string(type_t type, raw_t data, str &name) {
    prop   *p;
    u8     *p_value = get_member_address(type, data, name, p);
    if (!p_value)
        return null;
    memory *m       = p->member_type->functions->to_string(p_value);
    return  m;
}

memory  *mx::to_string() const {
    if      (mem->type == typeof(i8) ) return memory::string(std::to_string(*(i8*)  mem->origin));
    else if (mem->type == typeof(i16)) return memory::string(std::to_string(*(i16*) mem->origin));
    else if (mem->type == typeof(i32)) return memory::string(std::to_string(*(i32*) mem->origin));
    else if (mem->type == typeof(i64)) return memory::string(std::to_string(*(i64*) mem->origin));
    else if (mem->type == typeof(u8) ) return memory::string(std::to_string(*(u8*)  mem->origin));
    else if (mem->type == typeof(u16)) return memory::string(std::to_string(*(u16*) mem->origin));
    else if (mem->type == typeof(u32)) return memory::string(std::to_string(*(u32*) mem->origin));
    else if (mem->type == typeof(u64)) return memory::string(std::to_string(*(u64*) mem->origin));
    else if (mem->type == typeof(r32)) return memory::string(std::to_string(*(r32*) mem->origin));
    else if (mem->type == typeof(r64)) return memory::string(std::to_string(*(r64*) mem->origin));
    else if (mem->type == typeof(bool))return memory::string(std::to_string(*(bool*)mem->origin));

    else if  ((mem->type->traits & traits::enum_primitive) && mem->type->ref) {
        int iraw = *(int*)mem->origin;
        memory *res = mem->type->ref->lookup(u64(iraw));
        return res;
    }
    else if   (mem->type->functions->to_string)
        return mem->type->functions->to_string(mem->origin); /// call to_string() on context class
    
    else   if (mem->type->schema &&
                mem->type->schema->bind->data->functions &&
                mem->type->schema->bind->data->functions->to_string)
        return mem->type->schema->bind->data->functions->to_string(mem->origin); /// or data...
    
    else if (mem->type == typeof(char))
        return mem->grab();
    
    else {
        type_t id = mem->type;
        static char buf[128];
        const int l = snprintf(buf, sizeof(buf), "%s/%p", id->name, (void*)mem);
        return memory::stringify(cstr(buf), l);
    }
}

}