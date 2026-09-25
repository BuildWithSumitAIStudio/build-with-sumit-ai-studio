from DataIngestion.LocalDocumentQuery.pdfdataquery import PDFDataQuery

response = PDFDataQuery.pdf_data_query("simple-pdf-index",
                                       "Which employees are eligible for work from home and what document they need to provide?")

print(response)