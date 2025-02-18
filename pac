var FindProxyForURL = function(init, profiles) {
    return function(url, host) {
        "use strict";
        var result = init, scheme = url.substr(0, url.indexOf(":"));
        do {
            result = profiles[result];
            if (typeof result === "function") result = result(url, host, scheme);
        } while (typeof result !== "string" || result.charCodeAt(0) === 43);
        return result;
    };
}("+auto switch", {
    "+auto switch": function(url, host, scheme) {
        "use strict";
        if (/(?:^|\.)github\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)raw\.githubusercontent\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)cookielaw\.org$/.test(host)) return "+proxy";
        if (/(?:^|\.)segment\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)6sc\.co$/.test(host)) return "+proxy";
        if (/(?:^|\.)marketo\.net$/.test(host)) return "+proxy";
        if (/(?:^|\.)googletagmanager\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)dockerstatic\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)docker\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)djeqr6to3dedg\.cloudfront\.net$/.test(host)) return "+proxy";
        if (/(?:^|\.)gravatar\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)jnn-pa\.googleapis\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)gstatic\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)youtube\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)ytimg\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)jsdelivr\.net$/.test(host)) return "+proxy";
        if (/(?:^|\.)oaistatic\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)github\.io$/.test(host)) return "+proxy";
        if (/(?:^|\.)v2raya\.org$/.test(host)) return "+proxy";
        if (/(?:^|\.)v2ex\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)avatars\.githubusercontent\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)wsj\.net$/.test(host)) return "+proxy";
        if (/(?:^|\.)googleusercontent\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)google\.com$/.test(host)) return "+proxy";
        if (/(?:^|\.)openai\.com$/.test(host)) return "+proxy";
        if (/^127\.0\.0\.1$/.test(host)) return "DIRECT";
        return "DIRECT";
    },
    "+proxy": function(url, host, scheme) {
        "use strict";
        if (/^127\.0\.0\.1$/.test(host) || /^::1$/.test(host) || /^localhost$/.test(host)) return "DIRECT";
        return "SOCKS5 127.0.0.1:10808; SOCKS 127.0.0.1:10808";
    }
});
