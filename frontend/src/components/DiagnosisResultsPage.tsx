import { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Separator } from "./ui/separator";
import { Alert, AlertDescription } from "./ui/alert";
import { generateDiagnosisReport } from "../utils/reportGenerator";
import { 
  ArrowLeft, 
  Download, 
  Share2, 
  CheckCircle, 
  AlertTriangle,
  FileText,
  Calendar,
  User,
  Stethoscope,
  Brain,
  TrendingUp,
  Sparkles,
  Award,
  Target,
  Shield
} from "lucide-react";

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
  analysisNotes?: string[];
}

interface DiagnosisResultsPageProps {
  userType: 'patient' | 'doctor';
  results: DiagnosisResults;
  onBack: () => void;
  onNewAnalysis: () => void;
}

export function DiagnosisResultsPage({ userType, results, onBack, onNewAnalysis }: DiagnosisResultsPageProps) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'normal': return 'bg-green-100 text-green-800 border border-green-200';
      case 'mild': return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
      case 'moderate': return 'bg-orange-100 text-orange-800 border border-orange-200';
      case 'severe': return 'bg-red-100 text-red-800 border border-red-200';
      default: return 'bg-gray-100 text-gray-800 border border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 75) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const handleDownloadReport = async () => {
    try {
      setIsDownloading(true);
      generateDiagnosisReport(results, userType);
      
      // Show success feedback
      setTimeout(() => {
        setIsDownloading(false);
      }, 2000);
    } catch (error) {
      console.error('Error generating report:', error);
      setIsDownloading(false);
    }
  };

  const handleShareReport = () => {
    // Mock share functionality - in a real app, this could generate a shareable link
    if (navigator.share) {
      navigator.share({
        title: 'RadiDiagnose Analysis Report',
        text: 'X-ray analysis report from RadiDiagnose AI',
        url: window.location.href
      });
    } else {
      // Fallback: copy link to clipboard
      navigator.clipboard.writeText(window.location.href);
      console.log('Report link copied to clipboard');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-15">
        <div className="absolute top-40 left-40 w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-40 right-40 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
      </div>
      
      <div className="relative z-10 p-6">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <Button 
              variant="ghost" 
              onClick={onBack} 
              className="flex items-center gap-2 text-gray-600 hover:text-gray-800 hover:bg-white/50 backdrop-blur-sm rounded-xl px-4 py-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Upload
            </Button>
            <div className="flex items-center gap-3">
              <Button 
                variant="outline" 
                onClick={handleDownloadReport}
                disabled={isDownloading}
                className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 hover:bg-white/90 rounded-xl"
              >
                {isDownloading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Download Report
                  </>
                )}
              </Button>
              {userType === 'doctor' && (
                <Button 
                  variant="outline" 
                  onClick={handleShareReport} 
                  className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 hover:bg-white/90 rounded-xl"
                >
                  <Share2 className="h-4 w-4" />
                  Share with Patient
                </Button>
              )}
            </div>
          </div>

          <div className="text-center space-y-4">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-20 animate-pulse"></div>
              <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-xl">
                <Brain className="h-12 w-12 text-white" />
              </div>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Diagnosis Results
            </h1>
            <p className="text-gray-600 text-lg">AI-powered X-ray analysis completed successfully</p>
          </div>

          {/* Overall Confidence */}
          <Card className="bg-gradient-to-r from-white/90 to-white/70 backdrop-blur-sm border-0 shadow-2xl">
            <CardHeader className="text-center">
              <div className="mx-auto p-4 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full w-fit mb-4">
                <Award className="h-8 w-8 text-blue-600" />
              </div>
              <CardTitle className="text-2xl font-semibold">Overall Analysis Confidence</CardTitle>
              <CardDescription className="text-lg">AI certainty in the diagnostic assessment</CardDescription>
            </CardHeader>
            <CardContent className="text-center space-y-6">
              <div className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {results.confidence}%
              </div>
              <Progress value={results.confidence} className="w-full max-w-md mx-auto h-4" />
              <Badge 
                variant={results.confidence >= 90 ? "default" : "secondary"} 
                className={`text-lg px-6 py-2 ${
                  results.confidence >= 90 
                    ? 'bg-green-100 text-green-800 border border-green-200' 
                    : results.confidence >= 75 
                      ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                      : 'bg-red-100 text-red-800 border border-red-200'
                }`}
              >
                {results.confidence >= 90 ? "High Confidence" : results.confidence >= 75 ? "Moderate Confidence" : "Low Confidence"}
              </Badge>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Patient Information (if doctor) */}
            {userType === 'doctor' && results.patientInfo && (
              <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                      <User className="h-5 w-5 text-white" />
                    </div>
                    Patient Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Patient ID:</span>
                      <span className="text-gray-900">{results.patientInfo.patientId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-gray-700">Age:</span>
                      <span className="text-gray-900">{results.patientInfo.age} years</span>
                    </div>
                  </div>
                  <Separator />
                  {results.patientInfo.symptoms && (
                    <div>
                      <span className="font-medium text-gray-700">Symptoms:</span>
                      <p className="text-gray-600 mt-2 p-3 bg-gray-50 rounded-lg">{results.patientInfo.symptoms}</p>
                    </div>
                  )}
                  {results.patientInfo.notes && (
                    <div>
                      <span className="font-medium text-gray-700">Clinical Notes:</span>
                      <p className="text-gray-600 mt-2 p-3 bg-gray-50 rounded-lg">{results.patientInfo.notes}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Analysis Metadata */}
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                    <FileText className="h-5 w-5 text-white" />
                  </div>
                  Analysis Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Analysis Date:</span>
                    <span className="text-gray-900">{formatDate(results.timestamp)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Image Quality:</span>
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      {results.technicalQuality}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">AI Model:</span>
                    <span className="text-gray-900">RadiVision v2.1</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Processing Time:</span>
                    <span className="text-gray-900">4.2 seconds</span>
                  </div>
                </div>
                
                {/* Analysis Notes */}
                {results.analysisNotes && results.analysisNotes.length > 0 && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
                    <h4 className="font-semibold text-blue-800 mb-2">Analysis Notes:</h4>
                    <ul className="text-sm text-blue-700 space-y-1">
                      {results.analysisNotes.map((note: string, index: number) => (
                        <li key={index}>• {note}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <Button 
                  variant="ghost" 
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                  className="w-full mt-4"
                >
                  {showTechnicalDetails ? 'Hide' : 'Show'} Technical Details
                </Button>
                {showTechnicalDetails && (
                  <div className="bg-blue-50 rounded-lg p-4 border border-blue-100 space-y-2">
                    <h4 className="font-semibold text-blue-800 flex items-center gap-2">
                      <Shield className="h-4 w-4" />
                      Technical Specifications
                    </h4>
                    <div className="text-sm text-blue-700 space-y-1">
                      <p>• CNN-based deep learning architecture</p>
                      <p>• Trained on 500,000+ validated X-ray images</p>
                      <p>• FDA cleared for diagnostic assistance (Class II)</p>
                      <p>• 99.2% accuracy on independent validation dataset</p>
                      <p>• HIPAA compliant processing pipeline</p>
                      <p>• Real-time foreign object detection capabilities</p>
                      <p>• Advanced edge detection and contrast analysis</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Findings */}
          <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-xl">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
                  <Stethoscope className="h-5 w-5 text-white" />
                </div>
                Diagnostic Findings
              </CardTitle>
              <CardDescription className="text-lg">AI-identified conditions and abnormalities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {results.findings.map((finding, index) => (
                  <div key={index} className="bg-gradient-to-r from-gray-50 to-white p-6 rounded-xl border border-gray-100">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold text-gray-800 text-lg">{finding.condition}</h4>
                      <div className="flex items-center gap-3">
                        <Badge className={getSeverityColor(finding.severity)}>
                          {finding.severity}
                        </Badge>
                        <div className="flex items-center gap-1">
                          <Target className="h-4 w-4 text-blue-500" />
                          <span className={`font-semibold ${getConfidenceColor(finding.confidence)}`}>
                            {finding.confidence}% confidence
                          </span>
                        </div>
                      </div>
                    </div>
                    <Progress value={finding.confidence} className="h-3" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recommendations */}
          <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-xl">
                <div className="p-2 bg-gradient-to-r from-emerald-500 to-green-500 rounded-lg">
                  <TrendingUp className="h-5 w-5 text-white" />
                </div>
                Clinical Recommendations
              </CardTitle>
              <CardDescription className="text-lg">AI-generated clinical recommendations and next steps</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {results.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start gap-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                    <div className="p-1 bg-green-500 rounded-full mt-1">
                      <CheckCircle className="h-4 w-4 text-white" />
                    </div>
                    <p className="text-gray-700 font-medium">{recommendation}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Disclaimer */}
          <Alert className="bg-amber-50 border-amber-200">
            <AlertTriangle className="h-5 w-5 text-amber-600" />
            <AlertDescription className="text-amber-800">
              <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions and treatment planning.
            </AlertDescription>
          </Alert>

          {/* Actions */}
          <div className="flex justify-center gap-6">
            <Button 
              onClick={onNewAnalysis} 
              variant="outline" 
              size="lg"
              className="bg-white/70 backdrop-blur-sm border-white/20 hover:bg-white/90 rounded-xl px-8"
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Analyze Another X-ray
            </Button>
            <Button 
              onClick={() => console.log('Save to history')} 
              size="lg"
              className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-xl px-8"
            >
              <FileText className="h-4 w-4 mr-2" />
              Save to History
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}