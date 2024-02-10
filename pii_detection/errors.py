class TokenWithMultipleLabelsError(Exception):
    def __init__(self, token, labels, document_id=None):
        self.token = token
        self.labels = labels
        self.message = f"Token '{token}' has multiple labels: {labels}"
        if document_id:
            self.message += f" in document {document_id}"
        super().__init__(self.message)
