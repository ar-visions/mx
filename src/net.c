
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <windows.h>
#endif

#include <mbedtls/build_info.h>
#include <mbedtls/platform.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/x509.h>
#include <mbedtls/ssl.h>
#include <mbedtls/net_sockets.h>
#include <mbedtls/error.h>
#include <mbedtls/debug.h>
//#include "test/certs.h"

#if defined(MBEDTLS_SSL_CACHE_C)
#include <mbedtls/ssl_cache.h>
#endif

#ifndef WIN32
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <mbedtls/mbedtls_config.h> /// always must include first!
#include <mbedtls/net_sockets.h>
#include <mbedtls/ssl.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/error.h>
#include <mbedtls/debug.h>

#include <import>

#define len(I,...) ({ __typeof__(I) _i_ = I; _i_ ? (num)ftableI(_i_)->len(_i_, ## __VA_ARGS__) : (num)0; })

void test1() {
    string s = string("hi");
    int l = len(s);
}

// Implementation example:
none Session_with_TLS(Session s, TLS tls) {
    mbedtls_ssl_init(&s->ssl);
    mbedtls_net_init(&s->fd);
    mbedtls_ssl_setup(&s->ssl, &tls->conf);
}

none Session_with_uri(Session s, uri addr) {
    s->tls = TLS(addr, addr); 
}

bool Session_bind(Session s, uri addr) {
    string s_port = format("%i", addr->port);
    i32 res = mbedtls_net_bind(&s->fd, addr->host->chars, s_port->chars, MBEDTLS_NET_PROTO_TCP);
    if (res != 0) {
        print("mbedtls_net_bind: fails with %i", res);
        return false;
    }
    return true;
}

bool Session_connect(Session s) {
    string host = s->tls->url->host;
    i32 port = s->tls->url->port;
    
    i32 ret = mbedtls_ssl_setup(&s->ssl, &s->tls->conf);
    if (ret != 0) {
        error("mbedtls_ssl_setup failed: %i", ret);
        return false;
    }

    string str_port = format("%i", port);
    ret = mbedtls_ssl_set_hostname(&s->ssl, host->chars);
    if (ret != 0) {
        error("mbedtls_ssl_set_hostname failed: %i", ret);
        return false;
    }
    
    ret = mbedtls_net_connect(&s->fd, host->chars, str_port->chars, MBEDTLS_NET_PROTO_TCP);
    if (ret != 0) {
        error("mbedtls_net_connect failed: %i", ret);
        return false;
    }
    
    mbedtls_ssl_set_bio(&s->ssl, &s->fd, mbedtls_net_send, mbedtls_net_recv, null);
    
    while ((ret = mbedtls_ssl_handshake(&s->ssl)) != 0) {
        if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
            error("mbedtls_ssl_handshake failed: %i", ret);
            return false;
        }
    }
    s->connected = true;
    return true;
}

bool Session_close(Session s) {
    i32 ret;
    while ((ret = mbedtls_ssl_close_notify(&s->ssl)) < 0) {
        if (ret != MBEDTLS_ERR_SSL_WANT_READ && 
            ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
            print("mbedtls_ssl_close_notify returned %i", ret);
            return false;
        }
    }
    return true;
}

none Session_set_timeout(Session s, i64 t) {
    s->timeout_ms = t;
}

bool Session_read_sz(Session s, handle v, sz sz) {
    i32 st = 0;
    for (i32 len = sz; len > 0;) {
        i32 rcv = mbedtls_ssl_read(&s->ssl, v + st, len);
        if (rcv <= 0)
            return false;
        len -= rcv;
        st  += rcv;
    }
    return true;
}

sz Session_recv(Session s, handle buf, sz len) {
    sz sz;
    do {
        sz = mbedtls_ssl_read(&s->ssl, buf, len);
        if (sz == MBEDTLS_ERR_SSL_WANT_READ || sz == MBEDTLS_ERR_SSL_WANT_WRITE)
            continue;
        break;
    } while(1);
    return sz;
}

sz Session_send(Session s, handle buf, sz len) {
    sz ret;
    while ((ret = mbedtls_ssl_write(&s->ssl, buf, len)) <= 0) {
        if (ret == MBEDTLS_ERR_NET_CONN_RESET)
            return 0;
        if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE)
            break;
    }
    return ret;
}

sz Session_send_string(Session s, string v) {
    return send(s, v->chars, v->len);
}

vector Session_read_until(Session s, string match, i32 max_len) {
    vector rbytes = new(vector);
    sz slen = match->len;
    
    for (;;) {
        push(rbytes, 0);
        sz sz = len(rbytes);
        if (!recv(s, &((char*)data(rbytes))[sz - 1], 1))
            return null;
            
        if (sz >= slen && 
            memcmp(&((char*)data(rbytes))[sz - slen], match->chars, slen) == 0)
            break;
            
        if (sz == max_len)
            return null;
    }
    return rbytes;
}

Session Session_accept(TLS tls) {
    Session client = Session(tls);
    
    for (;;) {
        mbedtls_net_init(&client->fd);
        mbedtls_ssl_setup(&client->ssl, &client->tls->conf);

        i32 ret;
        if ((ret = mbedtls_net_accept(&tls->fd, &client->fd, null, 0, null)) != 0) {
            return null;
        }
        mbedtls_ssl_session_reset(&client->ssl);
        
        bool retry = false;
        mbedtls_ssl_set_bio(&client->ssl, &client->fd, 
                           mbedtls_net_send, mbedtls_net_recv, null);
        while ((ret = mbedtls_ssl_handshake(&client->ssl)) != 0) {
            if (ret != MBEDTLS_ERR_SSL_WANT_READ && 
                ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
                print("mbedtls_ssl_handshake: %i", ret);
                retry = true;
                break;
            }
        }
        if (!retry)
            break;
    }
    client->connected = true;
    return client;
}


void mbedtls_debug(void *ctx, int level, const char *file, int line, const char *str) {
    ((void) level);
    fprintf((FILE *) ctx, "mbedtls: %s:%04d: %s", file, line, str);
    fflush((FILE *) ctx);
}

none TLS_init(TLS tls) {
    static bool init_done = false;
    if (!init_done) {
        #ifdef _WIN32
        WSADATA wsa_data;
        i32 wsa = WSAStartup(MAKEWORD(2,2), &wsa_data);
        if (wsa != 0) {
            print("(sock) WSAStartup failed: %i", wsa);
            return;
        }
        #endif
        init_done = true;
    }

    /*
    tls->fd         = (mbedtls_net_context *)     calloc(1, sizeof(mbedtls_net_context));
    tls->conf       = (mbedtls_ssl_config *)      calloc(1, sizeof(mbedtls_ssl_config));
    tls->srvcert    = (mbedtls_x509_crt *)        calloc(1, sizeof(mbedtls_x509_crt));
    tls->pkey       = (mbedtls_pk_context *)      calloc(1, sizeof(mbedtls_pk_context));
    tls->entropy    = (mbedtls_entropy_context *) calloc(1, sizeof(mbedtls_entropy_context));
    tls->ctr_drbg   = (mbedtls_ctr_drbg_context *)calloc(1, sizeof(mbedtls_ctr_drbg_context));
    */

    mbedtls_net_init(&tls->fd);
    mbedtls_ssl_config_init(&tls->conf);
    mbedtls_x509_crt_init(&tls->srvcert);
    mbedtls_pk_init(&tls->pkey);
    mbedtls_entropy_init(&tls->entropy);
    mbedtls_ctr_drbg_init(&tls->ctr_drbg);

    print("  . Seeding the random number generator...");
    static string pers;
    if (!pers) pers = string("A-type::net");
    i32 ret = mbedtls_ctr_drbg_seed(&tls->ctr_drbg, mbedtls_entropy_func, &tls->entropy,
                                   (handle)pers->chars, strlen(pers));
    if (ret != 0) {
        print(" failed\n  ! mbedtls_ctr_drbg_seed returned %i", ret);
        return;
    }

    print(" ok\n");
    print("\n  . Loading the server cert. and key...");

    string host = tls->url->host;
    string pub = format("ssl/%s.crt", host->chars);
    string prv = format("ssl/%s.key", host->chars);

    ret = mbedtls_x509_crt_parse_file(&tls->srvcert, pub->chars);
    if (ret != 0) {
        print("mbedtls_x509_crt_parse returned %i\n", ret);
        return;
    }

    ret = mbedtls_pk_parse_keyfile(&tls->pkey, prv->chars, 0, mbedtls_ctr_drbg_random, &tls->ctr_drbg);
    if (ret != 0) {
        print("mbedtls_pk_parse_key returned %i\n", ret);
        return;
    }

    string port = format("%i", tls->url->port);
    ret = mbedtls_net_bind(&tls->fd, host->chars, port->chars, MBEDTLS_NET_PROTO_TCP);
    if (ret != 0) {
        print("mbedtls_net_bind returned %i\n", ret);
        return;
    }

    ret = mbedtls_ssl_config_defaults(&tls->conf,
                                     MBEDTLS_SSL_IS_SERVER,
                                     MBEDTLS_SSL_TRANSPORT_STREAM,
                                     MBEDTLS_SSL_PRESET_DEFAULT);
    if (ret != 0) {
        print("mbedtls_ssl_config_defaults returned %i\n", ret);
        return;
    }

    mbedtls_ssl_conf_rng(&tls->conf, mbedtls_ctr_drbg_random, &tls->ctr_drbg);
    mbedtls_ssl_conf_dbg(&tls->conf, mbedtls_debug, stdout);
    mbedtls_ssl_conf_ca_chain(&tls->conf, tls->srvcert.next, null);
    
    ret = mbedtls_ssl_conf_own_cert(&tls->conf, &tls->srvcert, &tls->pkey);
    if (ret != 0) {
        print("mbedtls_ssl_conf_own_cert returned %i\n", ret);
        return;
    }
}


message message_with_sock(message m, sock sc) {
    if (read_headers(m, sc)) {
        read_content(m, sc);
        string status = get(m->headers, string("Status"));
        m->code       = atoi(status->chars);
    }
    return m;
}


message message_with_i32(message m, i32 code) {
    m->code = code;
    return m;
}

message message_with_string(message m, string text) {
    m->content = text;
    m->code = 200;
    return m;
}

message message_with_path(message m, path p, object modified_since) {
    verify(exists(p) == Exists_file, "path must exist");
    string content = cast(string, read(p, typeid(string)));
    m->content = content;
    string mime = format("text/plain"); // TODO: implement mime_type
    set(m->headers, str("Content-Type"), mime);
    m->code = 200;
    return m;
}

message message_with_content(message m, object content, map headers, uri query) {
    m->query = query;
    m->headers = headers;
    m->content = content;
    m->code = 200;
    return m;
}

web message_method_type(message m) {
    return m->query->mtype;
}


bool message_read_headers(message m, sock sc) {
    i32 line = 0;
    for (;;) {
        vector rbytes = read_until(sc, str("\r\n"), 8192);
        sz sz = len(rbytes);
        if (sz == 0)
            return false;
        
        if (sz == 2)
            break;

        if (line++ == 0) {
            string hello = new(string, chars, data(rbytes), ref_length, sz - 2);
            array sp = split(hello, str(" "));
            
            // Handle response or request line
            if (len(hello) >= 12) {
                if (len(sp) >= 3) {
                    m->query = hello;
                    set(m->headers, string("Status"), get(sp, 1));
                }
            }
        } else {
            for (int i = 0; i < sz; i++) {
                if (((char*)data(rbytes))[i] == ':') {
                    string k = new(string, chars, data(rbytes), ref_length, i);
                    string v = new(string, chars, &((char*)data(rbytes))[i + 2], 
                                ref_length, sz - len(k) - 4);
                    set(m->headers, k, v);
                    break;
                }
            }
        }
    }
    return true;
}

bool message_read_content(message m, sock sc) {
    string te = str("Transfer-Encoding");
    string cl = str("Content-Length");
    string ce = str("Content-Encoding");
    
    string encoding = contains(m->headers, te) ? get(m->headers, ce) : null;
    i32 clen = -1;

    object o = get(m->headers, cl);
    if (o) {
        string v = instanceof(o, typeid(string));
        if (v) {
            clen = atoi(v->chars);
        } else {
            print("unsupported len format: %s", isa(o)->name);
        }
    }
    
    bool chunked = encoding && strcmp(get(m->headers, te), "chunked") == 0;
    num content_len = clen;
    num rlen = 0;
    const num r_max = 1024;
    bool error = false;
    num iter = 0;
    vector v_data = new(vector);

    verify(!(clen >= 0 && chunked), "invalid transfer encoding");

    if (!chunked && clen <= 0) {
        m->content = null;
        return true;
    }

    if (!(!chunked && clen == 0)) {
        do {
            if (chunked) {
                if (iter++ > 0) {
                    char crlf[2];
                    if (!read_sz(sc, crlf, 2) || memcmp(crlf, "\r\n", 2) != 0) {
                        error = true;
                        break;
                    }
                }
                vector rbytes = read_until(sc, str("\r\n"), 64);
                if (!rbytes) {
                    error = true;
                    break;
                }
                
                // Parse hex length
                content_len = strtol(data(rbytes), null, 16);
                if (content_len == 0)
                    break;
            }

            bool sff = content_len == -1;
            for (num rcv = 0; sff || rcv < content_len; rcv += rlen) {
                num rx = min(r_max, content_len - rcv);
                char buf[r_max];
                rlen = recv(sc, buf, rx);
                
                if (rlen > 0)
                    concat(v_data, buf, rlen);
                else if (rlen < 0) {
                    error = !sff;
                    break;
                } else if (rlen == 0) {
                    error = true;
                    break;
                }
            }
        } while (!error && chunked && content_len != 0);
    }

    if (!error) {
        string ctype = contains(m->headers, "Content-Type") ? 
                      get(m->headers, str("Content-Type")) : null;

        if (ctype && starts_with(ctype, str("application/json"))) {
            if (encoding) {
                if (strcmp(encoding, "gzip") == 0) {
                    // TODO: implement inflate
                    v_data = null;
                }
            }
            
            // Parse JSON
            m->content = null; // TODO: implement JSON parsing
        } else if (ctype && starts_with(ctype, str("text/"))) {
            m->content = new(string, chars, data(v_data), ref_length, len(v_data));
        } else {
            verify(len(v_data) == 0, "unsupported content type");
            m->content = null;
        }
    }

    return !error;
}


/// query/request construction
message message_query(uri server, map headers, object content) {
    message m;
    m->query   = uri(
        mtype,web_Get, proto,server->proto, host,server->host,
        port,server->port, query,server->query,
        resource,server->resource, args,server->args,
        version,server->version);
    m->headers = headers;
    m->content = content;
    return m;
}

/// response construction, uri is not needed
message message_response(uri query, i32 code, object content, map headers) {
    message r;
    r->query    = uri(
        mtype,web_Response, proto,query->proto, host,query->host,
        port,query->port, query,query->query,
        resource,query->resource, args,query->args,
        version,query->version);
    r->code     = code;
    r->headers  = headers;
    r->content  = content;
    return r;
}

symbol code_symbol(i32 code) {
    static map symbols = null;
    if (!symbols) {
        symbols = new(map);
        set(symbols, i(200), string("OK"));
        set(symbols, i(201), string("Created"));
        set(symbols, i(202), string("Accepted"));
        set(symbols, i(203), string("Non-Authoritative Information"));
        set(symbols, i(204), string("No Content"));
        set(symbols, i(205), string("Reset Content"));
        set(symbols, i(206), string("Partial Content"));
        set(symbols, i(300), string("Multiple Choices"));
        set(symbols, i(301), string("Moved Permanently"));
        set(symbols, i(302), string("Found"));
        set(symbols, i(303), string("See Other"));
        set(symbols, i(304), string("Not Modified"));
        set(symbols, i(307), string("Temporary Redirect"));
        set(symbols, i(308), string("Permanent Redirect"));
        set(symbols, i(400), string("Bad Request"));
        set(symbols, i(402), string("Payment Required"));
        set(symbols, i(403), string("Forbidden"));
        set(symbols, i(404), string("Not Found"));
        set(symbols, i(500), string("Internal Server Error"));
        set(symbols, i(0),   string("Unknown"));
    }
    string s_code = get(symbols, i(code));
    string result = s_code ? s_code : (string)get(symbols, i(0));
    return result->chars;
}

bool message_cast_bool(message m) {
    return m->query &&
           ((m->code >= 200 && m->code < 300) ||
            (m->code == 0 && (m->content || len(m->headers) > 0)));
}

string message_text(message m) {
    return cast(string, m->content);
}

map message_cookies(message m) {
    string cookies = get(m->headers, str("Set-Cookie"));
    if (!cookies)
        return new(map);

    string decoded = uri_decode(cookies);
    array parts = split(decoded, str(","));
    string all = get(parts, 0);
    array pairs = split(all, str(";"));
    map result = new(map);

    each(pairs, string, pair) {
        array kv = split(pair, str("="));
        if (len(kv) < 2)
            continue;
            
        string key = get(kv, 0);
        string val = get(kv, 1);
        set(result, key, val);
    }

    return result;
}


bool message_write_status(message m, sock sc) {
    string status = string("Status");
    i32 code = 0;
    object s = get(m->headers, status);
    if (s) {
        AType t = isa(s);
        int test = 1;
        test++;
        code = *(i32*)s;
    } else if (m->code)
        code = m->code;
    else
        code = 200;
    return send_object(sc, format("HTTP/1.1 %i %s\r\n", code, code_symbol(code)));
}


bool message_write_headers(message m, sock sc) {
    pairs(m->headers, ii) {
        string k = cast(string, ii->key);
        string v = cast(string, ii->value);
        if (strcmp(k->chars, "Status") == 0 || !v)
            continue;
        if (!send_object(sc, format("%o: %o\r\n", k, v)))
            return false;
    }
    return send_bytes(sc, "\r\n", 2);
}


string encode_fields(map fields) {
    if (!fields) 
        return str("");

    string post = new(string, alloc, 1024);
    bool first = true;

    pairs(fields, i) {
        string k = str(i->key);
        string v = str(i->value);
        
        if (!first) {
            append(post, "&");
        }
        string encoded = format("%s=%s", 
            uri_encode(k)->chars, 
            uri_encode(v)->chars);
        append(post, encoded);
        first = false;
    }
    
    return post;
}

bool message_write(message m, sock sc) {
    i32 ic = m->code;
    if (ic > 0) {
        symbol s = code_symbol(ic);
        verify(s, "invalid status code");
        string header = format("HTTP/1.1 %i %s\r\n", ic, s);
        if (!send_object(sc, header))
            return false;
    }

    if (m->content) {
        AType ct = isa(m->content);
        
        if (!contains(m->headers, "Content-Type") && 
            (ct == typeid(map) || ct == typeid(object)))
            set(m->headers, "Content-Type", "application/json");
            
        set(m->headers, "Connection", "keep-alive");
        
        if (contains(m->headers, "Content-Type") && 
            strcmp(get(m->headers, "Content-Type"), "application/json") == 0) {
            // TODO: JSON stringify
            return false;
        } else if (ct == typeid(map)) {
            string post = encode_fields(m->content);
            set(m->headers, "Content-Length", A_i64(len(post)));
            write_headers(m, sc);
            return send_object(sc, post);
        } else if (ct == typeid(u8)) {
            num byte_count = A_header(m->content)->count;
            set(m->headers, "Content-Length", A_i64(byte_count));
            return send_bytes(sc, m->content, byte_count);
        } else {
            verify(ct == typeid(string), "unsupported content type");
            set(m->headers, "Content-Length", A_i64(len((string)m->content)));
            write_headers(m, sc);
            return send_object(sc, m->content);
        }
    }
    
    set(m->headers, "Content-Length", "0");
    set(m->headers, "Connection", "keep-alive");
    return write_headers(m, sc);
}


string uri_addr(uri u) {
    return dns(u->host);
}

string dns(string hostname) {
    struct addrinfo hints = {0}, *res, *p;
    i32 status;
    char ip[INET6_ADDRSTRLEN];
    
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    status = getaddrinfo(hostname->chars, null, &hints, &res);
    if (status != 0) {
        print("DNS lookup failed: %i", status);
        return null;
    }

    string result = null;
    for (p = res; p != null; p = p->ai_next) {
        void* addr;
        if (p->ai_family == AF_INET) {
            struct sockaddr_in* ipv4 = (struct sockaddr_in*)p->ai_addr;
            addr = &(ipv4->sin_addr);
        } else {
            struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)p->ai_addr;
            addr = &(ipv6->sin6_addr);
        }

        char* ip_str = inet_ntop(p->ai_family, addr, ip, sizeof(ip));
        if (ip_str) {
            result = str(ip_str);
            break;
        }
    }

    freeaddrinfo(res);
    return result;
}

object request(uri url, map args) {
    map     st_headers   = new(map);
    object  null_content = null;
    map     headers      = contains(args, "headers") ? (map)get (args, "headers") : st_headers;
    object  content      = contains(args, "content") ? get (args, "content") : null_content;
    web     type         = contains(args, "method")  ? e_val(web, get(args, "method")) : web_Get;
    uri     query        = url;

    query->mtype = type;
    verify(query->mtype != web_undefined, "undefined web method type");

    sock client = sock(query);
    print("(net) request: %o", url);
    if (!connect(client))
        return null;

    // Send request line
    string method = estr(web, query->mtype);
    send_object(client, format("%o %o HTTP/1.1\r\n", method, query->query));

    // Default headers
    if (!contains(headers, "User-Agent"))      set(headers, "User-Agent", "ion:net");
    if (!contains(headers, "Accept"))          set(headers, "Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    if (!contains(headers, "Accept-Language")) set(headers, "Accept-Language", "en-US,en;q=0.9");
    if (!contains(headers, "Accept-Encoding")) set(headers, "Accept-Encoding", "gzip, deflate, br");
    if (!contains(headers, "Host"))            set(headers, "Host", query->host);

    message request = message(content, content, headers, headers, query, query);
    write(request, client);

    message response = message(client);
    close(client);

    return response;
}


uri uri_parse(string raw, uri ctx) {
    uri result = new(uri);
    array sp = split(raw, str(" "));
    bool has_method = len(sp) > 1;
    string lcase = len(sp) > 0 ? str(get(sp, 0)) : null;
    web m = e_val(web, has_method ? lcase->chars : "get");
    string u = get(sp, has_method ? 1 : 0);
    result->mtype = m;

    // find protocol separator
    num iproto = index_of(u, str("://"));
    if (iproto >= 0) {
        string p = mid(u, 0, iproto);
        u = mid(u, iproto + 3, len(u) - (iproto + 3));
        num ihost = index_of(u, str("/"));
        result->proto = e_val(protocol, p->chars);
        result->query = ihost >= 0 ? mid(u, ihost, len(u) - ihost) : str("/");
        string h = ihost >= 0 ? mid(u, 0, ihost) : u;
        num ih = index_of(h, str(":"));
        u = result->query;
        
        if (ih > 0) {
            result->host = mid(h, 0, ih);
            result->port = atoi(mid(h, ih + 1, len(h) - (ih + 1))->chars);
        } else {
            result->host = h;
            result->port = 0; // looked up by method
        }
    } else {
        // return default
        result->proto = ctx ? ctx->proto : protocol_undefined;
        result->host = ctx ? ctx->host : str("");
        result->port = ctx ? ctx->port : 0;
        result->query = u;
    }

    // parse resource and query
    num iq = index_of(u, str("?"));
    if (iq > 0) {
        result->resource = uri_decode(mid(u, 0, iq));
        string q = uri_decode(mid(u, iq + 1, len(u) - (iq + 1)));
        array a = split(q, str("&"));
        result->args = new(map);
        
        each(a, string, kv) {
            array sp = split(kv, str("="));
            object k = get(sp, 0);
            object v = len(sp) > 1 ? get(sp, 1) : k;
            set(result->args, k, v);
        }
    } else {
        result->resource = uri_decode(u);
    }

    if (len(sp) >= 3) {
        result->version = get(sp, 2);
    }

    return result;
}


string uri_encode(string s) {
    static string chars;
    if (!chars) chars = string(" -._~:/?#[]@!$&'()*+;%=");
    
    sz len = len(s);
    string v = string(alloc, len * 2);
    
    for (sz i = 0; i < len; i++) {
        char c = s->chars[i];
        bool a = ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'));
        if (!a)
            a = index_of(chars, string((i32)c)) != -1;
            
        if (!a) {
            append(v, "%");
            char hex[3];
            snprintf(hex, sizeof(hex), "%02x", (u8)c);
            append(v, hex);
        } else {
            char ch[2] = { c, 0 };
            append(v, ch);
        }
        
        if (c == '%')
            append(v, "%");
    }
    
    return v;
}

string uri_decode(string e) {
    num sz = len(e);
    string v = string(alloc, sz * 2);
    num i = 0;

    while (i < sz) {
        char c0 = e->chars[i];
        char cstr[2] = { c0, 0 };
        if (c0 == '%') {
            if (i >= sz - 1)
                break;
                
            char c1 = e->chars[i + 1];
            if (c1 == '%') {
                append(v, "%");
            } else {
                if (i >= sz - 2)
                    break;
                    
                char c2 = e->chars[i + 2];
                char hex[3] = {c1, c2, 0};
                u8 val;
                sscanf(hex, "%hhx", &val);
                char vstr[2] = { val, 0 };
                append(v, vstr);
                i += 2;
            }
        } else {
            append(v, (c0 == '+') ? " " : cstr);
        }
        i++;
    }
    
    return string(chars, data(v), ref_length, len(v));
}

// The handler function signature would be:
object handle_client(object target, object client_sock, object context) {
    sock s = client_sock;
    // Handle the client socket
    return A_bool(true);
}

object sock_listen(uri url, subprocedure handler) {
    TLS tls = TLS(url, url);
    
    for (;;) {
        sock client = sock_accept(tls);
        if (!client)
            break;

        // handler function will receive:
        // - target from handler->target
        // - client as the data arg 
        // - context from handler->ctx
        object result = invoke(handler, client);
        if (!cast(bool, result))
            break;
    }
    return tls;
}

void sock_with_uri(sock s, uri addr) {
    s->data = Session(TLS(addr, addr));
}

void sock_with_TLS(sock s, TLS tls) {
    s->data = Session(tls);
}

bool sock_bind(sock s, uri addr) {
    return Session_bind(s->data, addr);
}

bool sock_connect(sock s) {
    return Session_connect(s);
}

bool sock_close(sock s) {
    return Session_close(s->data);
}

none sock_set_timeout(sock s, i64 t) {
    Session_set_timeout(s->data, t);
}

bool sock_read_sz(sock s, handle v, sz sz) {
    return Session_read_sz(s->data, v, sz);
}

sz sock_recv(sock s, handle buf, sz len) {
    return Session_recv(s->data, buf, len);
}

sz sock_send_bytes(sock s, handle buf, sz len) {
    return Session_send(s->data, buf, len);
}

sz sock_send_object(sock s, object v) {
    string str = cast(string, v);
    return Session_send_string(s->data, str);
}

vector sock_read_until(sock s, string match, i32 max_len) {
    return Session_read_until(s->data, match, max_len);
}

sock sock_accept(TLS tls) {
    Session s = Session_accept(tls);
    return s ? sock(s->tls) : null;
}

bool sock_cast_bool(sock s) {
    return s->data->connected;
}

bool sock_read(sock s, handle buf, sz len) {
    sz actual = recv(s, buf, len);
    return actual == len;
}

// For JSON requests, success/failure handlers would have signatures like:
object on_success(object target, object response_data, object context) {
    // Handle successful JSON response
    return response_data;
}

object on_failure(object target, object error_data, object context) {
    // Handle failure
    return null;
}

object json_request(uri addr, map args, map headers, subprocedure success_handler, subprocedure failure_handler) {
    object response = request(addr, headers);
    
    if (!response) {
        return invoke(failure_handler, null);
    }

    if (isa(response) == typeid(map)) {
        return invoke(success_handler, response);
    } else {
        return invoke(failure_handler, response);
    }
}


define_class(uri,       A)
define_class(Session,   A)
define_class(TLS,       A)
define_class(sock,      A)
define_class(message,   A)

define_enum(web)
define_enum(protocol)
