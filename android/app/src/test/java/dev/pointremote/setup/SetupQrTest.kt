package dev.pointremote.setup

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SetupQrTest {
    @Test fun readsThePcAppsCode() {
        assertEquals(
            ServerSetup("http://192.168.1.20:8765", "a-b_c+d/e"),
            parseSetupQr("pointremote://setup?url=http%3A%2F%2F192.168.1.20%3A8765&key=a-b_c%2Bd%2Fe"),
        )
    }

    @Test fun ignoresOtherCodes() {
        assertNull(parseSetupQr("https://example.com"))
        assertNull(parseSetupQr("WIFI:S:home;P:secret;;"))
        assertNull(parseSetupQr("pointremote://other?url=http%3A%2F%2Fx&key=k"))
    }

    @Test fun knowsHomeNetworkAddresses() {
        listOf("192.168.1.20", "10.0.0.5", "172.20.1.1", "pc.local").forEach { assert(isPrivateHost(it)) { it } }
        listOf("8.8.8.8", "172.32.0.1", "example.com", "192.168.1").forEach { assert(!isPrivateHost(it)) { it } }
        val setup = parseSetupQr("pointremote://setup?url=http%3A%2F%2F203.0.113.9%3A8765&key=abcdefgh12")!!
        assert(!setup.isHomeNetwork)
    }

    @Test fun rejectsOddKeys() {
        assertNull(parseSetupQr("pointremote://setup?url=http%3A%2F%2F10.0.0.2&key=short"))
        assertNull(parseSetupQr("pointremote://setup?url=http%3A%2F%2F10.0.0.2&key=has%20spaces%20in%20it"))
    }

    @Test fun needsAUsableUrlAndKey() {
        assertNull(parseSetupQr("pointremote://setup?url=http%3A%2F%2F192.168.1.20%3A8765"))
        assertNull(parseSetupQr("pointremote://setup?url=ftp%3A%2F%2Fx&key=k"))
        assertNull(parseSetupQr("pointremote://setup?url=http%3A%2F%2F192.168.1.20&key=%20"))
        assertNull(parseSetupQr("pointremote://setup?url=not%20a%20url&key=k"))
    }
}
