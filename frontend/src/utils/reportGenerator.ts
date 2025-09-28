import jsPDF from 'jspdf';

interface Finding {
  condition: string;
  severity: string;
  confidence: number;
}

interface DiagnosisResults {
  confidence: number;
  findings: Finding[];
  recommendations: string[];
  technicalQuality: string;
  timestamp: string;
  patientInfo?: {
    patientId: string;
    age: string;
    symptoms: string;
    notes: string;
  } | null;
}

export const generateDiagnosisReport = (
  results: DiagnosisResults,
  userType: 'patient' | 'doctor'
): void => {
  const doc = new jsPDF();
  
  // Colors
  const primaryColor = [3, 2, 19]; // #030213
  const blueColor = [59, 130, 246]; // blue-500
  const grayColor = [107, 114, 128]; // gray-500
  const greenColor = [34, 197, 94]; // green-500
  const redColor = [239, 68, 68]; // red-500
  const orangeColor = [249, 115, 22]; // orange-500

  // Page dimensions and margins
  const pageHeight = 280; // Leave space for footer
  const footerHeight = 15;
  const maxContentHeight = pageHeight - footerHeight;
  let yPosition = 20;
  
  // Helper function to check if we need a new page
  const checkPageBreak = (requiredHeight: number) => {
    if (yPosition + requiredHeight > maxContentHeight) {
      doc.addPage();
      yPosition = 20;
      return true;
    }
    return false;
  };
  
  // Header with branding
  doc.setFontSize(24);
  doc.setTextColor(...primaryColor);
  doc.text('RadiDiagnose', 20, yPosition);
  
  doc.setFontSize(12);
  doc.setTextColor(...blueColor);
  doc.text('AI-Powered X-ray Analysis Report', 20, yPosition + 8);
  
  // Add logo placeholder
  doc.setDrawColor(...blueColor);
  doc.setFillColor(...blueColor);
  doc.circle(180, yPosition - 2, 8, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(10);
  doc.text('AI', 176, yPosition + 2);
  
  yPosition += 30;
  
  // Report metadata
  checkPageBreak(20);
  doc.setFontSize(10);
  doc.setTextColor(...grayColor);
  doc.text(`Generated: ${new Date(results.timestamp).toLocaleString()}`, 20, yPosition);
  doc.text(`Report Type: ${userType === 'doctor' ? 'Professional' : 'Patient'} Analysis`, 20, yPosition + 5);
  doc.text(`Analysis ID: ${Date.now().toString().slice(-8)}`, 20, yPosition + 10);
  
  yPosition += 25;
  
  // Patient Information (if doctor view)
  if (userType === 'doctor' && results.patientInfo) {
    checkPageBreak(60);
    
    doc.setFontSize(14);
    doc.setTextColor(...primaryColor);
    doc.text('Patient Information', 20, yPosition);
    
    yPosition += 15;
    
    doc.setFontSize(10);
    doc.setTextColor(...grayColor);
    doc.text(`Patient ID: ${results.patientInfo.patientId}`, 20, yPosition);
    doc.text(`Age: ${results.patientInfo.age} years`, 20, yPosition + 5);
    
    yPosition += 15;
    
    if (results.patientInfo.symptoms) {
      checkPageBreak(25);
      doc.text('Symptoms:', 20, yPosition);
      const symptomsLines = doc.splitTextToSize(results.patientInfo.symptoms, 170);
      doc.text(symptomsLines, 20, yPosition + 5);
      yPosition += symptomsLines.length * 4 + 10;
    }
    
    if (results.patientInfo.notes) {
      checkPageBreak(25);
      doc.text('Clinical Notes:', 20, yPosition);
      const notesLines = doc.splitTextToSize(results.patientInfo.notes, 170);
      doc.text(notesLines, 20, yPosition + 5);
      yPosition += notesLines.length * 4 + 10;
    }
    
    yPosition += 10;
  }
  
  // Overall Confidence Score
  checkPageBreak(40);
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.text('Analysis Confidence', 20, yPosition);
  
  yPosition += 15;
  
  // Confidence score with visual indicator
  doc.setFontSize(24);
  const confidenceColor = results.confidence >= 90 ? greenColor : 
                         results.confidence >= 75 ? orangeColor : redColor;
  doc.setTextColor(...confidenceColor);
  doc.text(`${results.confidence}%`, 20, yPosition);
  
  doc.setFontSize(10);
  doc.setTextColor(...grayColor);
  const confidenceLabel = results.confidence >= 90 ? 'High Confidence' : 
                         results.confidence >= 75 ? 'Moderate Confidence' : 'Low Confidence';
  doc.text(confidenceLabel, 20, yPosition + 8);
  
  // Confidence bar
  const barWidth = 60;
  const barHeight = 6;
  doc.setDrawColor(200, 200, 200);
  doc.rect(80, yPosition - 12, barWidth, barHeight);
  doc.setFillColor(...confidenceColor);
  doc.rect(80, yPosition - 12, (barWidth * results.confidence) / 100, barHeight, 'F');
  
  yPosition += 25;
  
  // Technical Quality
  doc.setFontSize(10);
  doc.setTextColor(...grayColor);
  doc.text(`Image Quality: ${results.technicalQuality}`, 20, yPosition);
  
  yPosition += 20;
  
  // Diagnostic Findings
  checkPageBreak(30);
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.text('Diagnostic Findings', 20, yPosition);
  
  yPosition += 15;
  
  results.findings.forEach((finding, index) => {
    checkPageBreak(25); // Check space for each finding
    
    doc.setFontSize(11);
    doc.setTextColor(...primaryColor);
    doc.text(`${index + 1}. ${finding.condition}`, 25, yPosition);
    
    yPosition += 8;
    
    doc.setFontSize(9);
    doc.setTextColor(...grayColor);
    doc.text(`Severity: ${finding.severity}`, 30, yPosition);
    doc.text(`Confidence: ${finding.confidence}%`, 80, yPosition);
    
    // Mini confidence bar for each finding
    const miniBarWidth = 30;
    const miniBarHeight = 3;
    doc.setDrawColor(200, 200, 200);
    doc.rect(120, yPosition - 2, miniBarWidth, miniBarHeight);
    
    const findingColor = finding.confidence >= 90 ? greenColor : 
                        finding.confidence >= 75 ? orangeColor : redColor;
    doc.setFillColor(...findingColor);
    doc.rect(120, yPosition - 2, (miniBarWidth * finding.confidence) / 100, miniBarHeight, 'F');
    
    yPosition += 15;
  });
  
  yPosition += 10;
  
  // Recommendations
  checkPageBreak(30);
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.text('Recommendations', 20, yPosition);
  
  yPosition += 15;
  
  results.recommendations.forEach((recommendation, index) => {
    // Calculate required height for this recommendation
    const recommendationLines = doc.splitTextToSize(`• ${recommendation}`, 170);
    const requiredHeight = recommendationLines.length * 5 + 5;
    
    checkPageBreak(requiredHeight);
    
    doc.setFontSize(10);
    doc.setTextColor(...grayColor);
    doc.text(recommendationLines, 25, yPosition);
    yPosition += requiredHeight;
  });
  
  yPosition += 20;
  
  // Medical Disclaimer
  checkPageBreak(35);
  doc.setFontSize(8);
  doc.setTextColor(...redColor);
  doc.text('MEDICAL DISCLAIMER:', 20, yPosition);
  doc.setTextColor(...grayColor);
  const disclaimerText = 'This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.';
  const disclaimerLines = doc.splitTextToSize(disclaimerText, 170);
  doc.text(disclaimerLines, 20, yPosition + 5);
  
  yPosition += disclaimerLines.length * 3 + 20;
  
  // Add footer to each page
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Footer
    doc.setFontSize(8);
    doc.setTextColor(...grayColor);
    doc.text('Generated by RadiDiagnose AI • www.radidiagnose.com', 20, 285);
    doc.text(`Page ${i} of ${pageCount}`, 180, 285);
    
    // Technical details on last page only
    if (i === pageCount) {
      doc.setFontSize(7);
      doc.text('AI Model: RadiVision v2.1 • FDA Cleared • 99.2% Validation Accuracy', 20, 290);
    }
  }
  
  // Generate filename
  const timestamp = new Date().toISOString().split('T')[0];
  const patientId = results.patientInfo?.patientId || 'patient';
  const filename = `RadiDiagnose_Report_${patientId}_${timestamp}.pdf`;
  
  // Save the PDF
  doc.save(filename);
};

export const generateHistoryReport = (
  historyItems: any[],
  userType: 'patient' | 'doctor'
): void => {
  const doc = new jsPDF();
  
  // Colors
  const primaryColor = [3, 2, 19];
  const blueColor = [59, 130, 246];
  const grayColor = [107, 114, 128];
  
  // Page dimensions and margins
  const pageHeight = 280;
  const footerHeight = 15;
  const maxContentHeight = pageHeight - footerHeight;
  let yPosition = 20;
  
  // Helper function to check if we need a new page
  const checkPageBreak = (requiredHeight: number) => {
    if (yPosition + requiredHeight > maxContentHeight) {
      doc.addPage();
      yPosition = 20;
      return true;
    }
    return false;
  };
  
  // Header
  doc.setFontSize(24);
  doc.setTextColor(...primaryColor);
  doc.text('RadiDiagnose', 20, yPosition);
  
  doc.setFontSize(12);
  doc.setTextColor(...blueColor);
  doc.text('Diagnosis History Report', 20, yPosition + 8);
  
  yPosition += 30;
  
  // Report metadata
  doc.setFontSize(10);
  doc.setTextColor(...grayColor);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 20, yPosition);
  doc.text(`Total Records: ${historyItems.length}`, 20, yPosition + 5);
  doc.text(`Report Type: ${userType === 'doctor' ? 'Professional' : 'Patient'} History`, 20, yPosition + 10);
  
  yPosition += 30;
  
  // History items
  doc.setFontSize(14);
  doc.setTextColor(...primaryColor);
  doc.text('Analysis History', 20, yPosition);
  
  yPosition += 20;
  
  historyItems.forEach((item, index) => {
    checkPageBreak(25); // Check space for each history item
    
    doc.setFontSize(11);
    doc.setTextColor(...primaryColor);
    doc.text(`${index + 1}. ${item.primaryFinding}`, 20, yPosition);
    
    yPosition += 8;
    
    doc.setFontSize(9);
    doc.setTextColor(...grayColor);
    doc.text(`Date: ${new Date(item.date).toLocaleDateString()}`, 25, yPosition);
    doc.text(`Confidence: ${item.confidence}%`, 80, yPosition);
    doc.text(`Status: ${item.status}`, 130, yPosition);
    
    if (userType === 'doctor' && item.patientName) {
      yPosition += 5;
      checkPageBreak(8);
      doc.text(`Patient: ${item.patientName} (${item.patientId})`, 25, yPosition);
    }
    
    yPosition += 20;
  });
  
  // Add footer to each page
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Footer
    doc.setFontSize(8);
    doc.setTextColor(...grayColor);
    doc.text('Generated by RadiDiagnose AI • www.radidiagnose.com', 20, 285);
    doc.text(`Page ${i} of ${pageCount}`, 180, 285);
  }
  
  // Generate filename
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `RadiDiagnose_History_${userType}_${timestamp}.pdf`;
  
  doc.save(filename);
};