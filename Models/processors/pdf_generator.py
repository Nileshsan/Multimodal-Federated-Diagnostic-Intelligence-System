import os
from pathlib import Path
from Models.processors.report_generator_with_pdf import StructuredReportGenerator
from datetime import datetime

def save_report_as_pdf(report, base_path=None):
    """
    Save the medical report as a PDF file in an organized directory structure.
    
    Args:
        report: The medical report object
        base_path: Optional custom base path for reports. If None, uses default structure.
        
    Returns:
        str: Path to the generated PDF file
    """
    # Set up directory structure
    if base_path is None:
        base_path = Path(os.getcwd()) / "reports" / "medical_reports"
    else:
        base_path = Path(base_path)
    
    # Create directory structure
    year_month = datetime.now().strftime("%Y/%m")
    report_dir = base_path / year_month
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp and diagnosis info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnosis = report.diagnostic_interpretation.get('primary_diagnosis', {}).get('condition', 'unspecified')
    diagnosis = "".join(c for c in diagnosis if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"medical_report_{timestamp}_{diagnosis}.pdf"
    
    # Create full output path
    output_path = report_dir / filename
    
    # Create report generator instance
    report_generator = StructuredReportGenerator()
    
    try:
        # Generate PDF
        report_generator.generate_pdf_report(report, str(output_path))
        print(f"\nPDF report generated successfully!")
        print(f"Location: {output_path}")
        
        # Create a simple index file in the month directory if it doesn't exist
        index_file = report_dir / "report_index.txt"
        if not index_file.exists():
            index_file.write_text("Medical Reports Index\n" + "="*20 + "\n\n")
            
        # Append to index
        with open(index_file, "a") as f:
            f.write(f"{timestamp} - {diagnosis}\n")
            f.write(f"File: {filename}\n")
            if report.diagnostic_interpretation.get('confidence_assessment', {}).get('confidence_level'):
                confidence = report.diagnostic_interpretation['confidence_assessment']['confidence_level']
                f.write(f"Confidence Level: {confidence}\n")
            f.write("-"*50 + "\n")
            
        return str(output_path)
        
    except Exception as e:
        print(f"\nError generating PDF report: {str(e)}")
        raise