class PDFLinkModel:
    def __init__(self, host, name, date) -> None:
        self.host = host
        self.name = name
        self.date = date

    def __str__(self) -> str:
        return(self.host)