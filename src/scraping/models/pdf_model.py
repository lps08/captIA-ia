class PDFLinkModel:
    def __init__(self, host, parent_host, name, created, date) -> None:
        self.host = host
        self.parent_host = parent_host
        self.name = name
        self.created = created
        self.date = date

    def __str__(self) -> str:
        return(self.host)