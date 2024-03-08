import ssl
import requests

class TLSAdapter(requests.adapters.HTTPAdapter):
    """
    Custom HTTPAdapter subclass with TLS configuration.
    """
    def init_poolmanager(self, *args, **kwargs):
        """
        Initialize the connection pool manager with custom SSL/TLS configuration.

        Overrides the 'init_poolmanager' method of the base class.
        Customizes SSL/TLS context to disable hostname verification, set ciphers, and adjust options.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            HTTPConnectionPool: A connection pool manager with customized SSL/TLS configuration.
        """
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        ctx.options |= 0x4
        kwargs["ssl_context"] = ctx
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)