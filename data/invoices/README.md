# Invoice Files Directory

Place your PDF invoice files in this directory for processing.

## Supported Formats
- PDF files (.pdf extension)
- Various invoice layouts and formats
- Multi-page invoices supported

## Sample Files
To test the system, you can:

1. **Use sample invoices** from the original repository
2. **Create test PDFs** with invoice-like content
3. **Use real invoice PDFs** (ensure they contain typical invoice information)

## Expected Invoice Content
The AI agents can extract:
- Invoice number
- Customer/vendor information
- Line items with quantities and prices
- Totals and subtotals
- Due dates
- Shipping information

## File Naming
- Use descriptive names: `invoice_001.pdf`, `customer_invoice_jan2024.pdf`
- Avoid special characters in filenames
- Keep filenames under 100 characters

## Processing Notes
- Files are processed in the order selected in the UI
- Large files may take longer to process
- The system creates backups of processing results in the `output/` directory

## Security
- Ensure invoices don't contain sensitive personal information beyond business data
- The system follows data retention policies defined in the configuration
- Audit trails are maintained for all processed files