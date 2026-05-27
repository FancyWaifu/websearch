"""SSRF guard tests for the MCP fetch tool.

`_url_safety_error` is the gate that stops an MCP client (or a prompt
injection delivered through a tool result) from steering this process at
cloud metadata, LAN hosts, or local services. Test the pure function — the
FastMCP wiring depends on the optional `mcp` package and isn't needed here.
"""
from unittest import mock

import pytest

from websearch.mcp_server import _url_safety_error


def _patch_dns(monkeypatch, ip: str) -> None:
    """Force socket.getaddrinfo to return `ip` for any host."""
    def fake(host, port, *a, **kw):
        family = 10 if ":" in ip else 2  # AF_INET6 vs AF_INET
        return [(family, 1, 6, "", (ip, port or 0))]
    monkeypatch.setattr("websearch.mcp_server.socket.getaddrinfo", fake)


def test_public_url_passes(monkeypatch):
    _patch_dns(monkeypatch, "93.184.216.34")  # example.com-ish
    assert _url_safety_error("https://example.com/article") is None


@pytest.mark.parametrize("scheme", ["file", "gopher", "ftp", "data", ""])
def test_non_http_schemes_blocked(scheme):
    url = f"{scheme}://example.com/x" if scheme else "example.com/x"
    err = _url_safety_error(url)
    assert err is not None
    assert "http" in err


@pytest.mark.parametrize("host", ["localhost", "LocalHost", "ip6-localhost"])
def test_localhost_names_blocked(host, monkeypatch):
    _patch_dns(monkeypatch, "127.0.0.1")
    err = _url_safety_error(f"http://{host}/x")
    assert err is not None
    assert "blocked" in err


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",        # loopback
        "10.0.0.1",         # RFC1918
        "172.16.5.5",       # RFC1918
        "192.168.50.155",   # the user's homelab range
        "169.254.169.254",  # AWS/GCP metadata
        "0.0.0.0",          # unspecified
        "::1",              # IPv6 loopback
        "fe80::1",          # IPv6 link-local
        "fc00::1",          # IPv6 ULA
        "224.0.0.1",        # multicast
    ],
)
def test_private_loopback_link_local_ips_blocked(ip, monkeypatch):
    _patch_dns(monkeypatch, ip)
    err = _url_safety_error("http://attacker-controlled.example/x")
    assert err is not None
    assert "non-public" in err


def test_bare_private_ip_in_url_blocked(monkeypatch):
    # No DNS lookup happens for a bare IP — getaddrinfo still returns it,
    # the check rejects via _ip_is_public regardless of path.
    _patch_dns(monkeypatch, "192.168.50.155")
    err = _url_safety_error("http://192.168.50.155:8080/admin")
    assert err is not None


def test_missing_host_blocked():
    err = _url_safety_error("http:///nopath")
    assert err is not None
    assert "host" in err.lower()


def test_dns_failure_blocked(monkeypatch):
    import socket as _socket
    def fail(*a, **kw):
        raise _socket.gaierror("nope")
    monkeypatch.setattr("websearch.mcp_server.socket.getaddrinfo", fail)
    err = _url_safety_error("https://does-not-resolve.example/x")
    assert err is not None
    assert "DNS" in err


def test_any_private_address_in_resolved_set_blocks(monkeypatch):
    """A multi-A record with one private answer should be refused — any one
    bad address is enough."""
    def fake(host, port, *a, **kw):
        return [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("192.168.50.155", 0)),
        ]
    monkeypatch.setattr("websearch.mcp_server.socket.getaddrinfo", fake)
    err = _url_safety_error("https://rebound.example/x")
    assert err is not None
