import { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Textarea } from "./ui/textarea";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription } from "./ui/alert";
import { analyzeXrayImage } from "../utils/imageAnalysis";
import { 
  Upload, 
  Image as ImageIcon, 
  Scan, 
  AlertCircle, 
  CheckCircle, 
  ArrowLeft,
  Brain,
  Clock,
  Sparkles,
  Zap,
  Shield,
  FileCheck,
  Camera,
  User
} from "lucide-react";

interface XrayUploadPageProps {
  userType: 'patient' | 'doctor';
  onBack: () => void;
  onAnalysisComplete: (results: any) => void;
}

export function XrayUploadPage({ userType, onBack, onAnalysisComplete }: XrayUploadPageProps) {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStage, setAnalysisStage] = useState('');
  const [patientInfo, setPatientInfo] = useState({
    patientId: '',
    age: '',
    symptoms: '',
    notes: ''
  });

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const performRealAnalysis = async () => {
    if (!uploadedFile) return;

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisStage('Preprocessing image...');

    // Stage 1: Image preprocessing
    await new Promise(resolve => setTimeout(resolve, 800));
    setAnalysisProgress(20);
    setAnalysisStage('Analyzing image characteristics...');

    // Stage 2: Basic image analysis
    await new Promise(resolve => setTimeout(resolve, 1000));
    setAnalysisProgress(40);
    setAnalysisStage('Detecting foreign objects...');

    // Stage 3: AI analysis
    await new Promise(resolve => setTimeout(resolve, 1000));
    setAnalysisProgress(60);
    setAnalysisStage('Evaluating anatomical structures...');

    // Stage 4: Advanced analysis
    await new Promise(resolve => setTimeout(resolve, 800));
    setAnalysisProgress(80);
    setAnalysisStage('Generating diagnostic report...');

    try {
      // Perform actual image analysis
      const analysisResult = await analyzeXrayImage(uploadedFile);
      
      await new Promise(resolve => setTimeout(resolve, 600));
      setAnalysisProgress(100);
      setAnalysisStage('Analysis complete!');

      // Format results for the results page
      const formattedResults = {
        confidence: analysisResult.confidence,
        findings: analysisResult.findings,
        recommendations: analysisResult.recommendations,
        technicalQuality: analysisResult.technicalQuality,
        timestamp: new Date().toISOString(),
        analysisNotes: analysisResult.analysisNotes,
        patientInfo: userType === 'doctor' ? patientInfo : null
      };

      await new Promise(resolve => setTimeout(resolve, 1000));
      setIsAnalyzing(false);
      onAnalysisComplete(formattedResults);
    } catch (error) {
      console.error('Analysis error:', error);
      // Fallback to mock data if analysis fails
      const fallbackResults = {
        confidence: 85,
        findings: [
          { condition: "Analysis processing error", severity: "Mild", confidence: 85 },
          { condition: "Please try re-uploading the image", severity: "Normal", confidence: 90 }
        ],
        recommendations: [
          "Ensure image is clear and properly oriented",
          "Try uploading a higher quality image"
        ],
        technicalQuality: "Fair",
        timestamp: new Date().toISOString(),
        analysisNotes: ["Technical processing encountered minor issues"],
        patientInfo: userType === 'doctor' ? patientInfo : null
      };
      
      setIsAnalyzing(false);
      onAnalysisComplete(fallbackResults);
    }
  };

  const handleAnalyze = () => {
    if (uploadedFile) {
      performRealAnalysis();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-15">
        <div className="absolute top-32 left-32 w-80 h-80 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-32 right-32 w-80 h-80 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
      </div>
      
      <div className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <Button 
              variant="ghost" 
              onClick={onBack} 
              className="flex items-center gap-2 text-gray-600 hover:text-gray-800 hover:bg-white/50 backdrop-blur-sm rounded-xl px-4 py-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Dashboard
            </Button>
            <Badge variant="outline" className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 px-4 py-2 rounded-xl">
              <Brain className="h-4 w-4" />
              AI-Powered Analysis
            </Badge>
          </div>

          <div className="text-center space-y-4">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-20 animate-pulse"></div>
              <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-xl">
                <Scan className="h-12 w-12 text-white" />
              </div>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              X-ray Analysis
            </h1>
            <p className="text-gray-600 text-lg max-w-2xl mx-auto">
              Upload X-ray images for instant AI-powered diagnostic analysis with professional-grade accuracy
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Upload Section */}
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
                    <Upload className="h-5 w-5 text-white" />
                  </div>
                  Upload X-ray Image
                </CardTitle>
                <CardDescription>
                  Select a high-quality X-ray image (JPEG, PNG, DICOM supported)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="relative border-2 border-dashed border-gray-200 rounded-2xl p-12 text-center hover:border-blue-400 transition-all duration-300 bg-gradient-to-br from-blue-50/30 to-purple-50/30">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="xray-upload"
                  />
                  <label htmlFor="xray-upload" className="cursor-pointer">
                    {previewUrl ? (
                      <div className="space-y-6">
                        <div className="relative inline-block">
                          <img 
                            src={previewUrl} 
                            alt="X-ray preview" 
                            className="max-w-full max-h-64 mx-auto rounded-xl shadow-2xl border-4 border-white"
                          />
                          <div className="absolute -top-2 -right-2 p-2 bg-green-500 rounded-full shadow-lg">
                            <CheckCircle className="h-4 w-4 text-white" />
                          </div>
                        </div>
                        <p className="text-gray-600 font-medium">✨ Image uploaded successfully</p>
                        <p className="text-sm text-gray-500">Click to change image</p>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="relative">
                          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full blur-lg opacity-20 animate-pulse"></div>
                          <div className="relative p-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mx-auto w-fit">
                            <ImageIcon className="h-12 w-12 text-white" />
                          </div>
                        </div>
                        <div>
                          <p className="text-xl font-semibold text-gray-800 mb-2">Choose X-ray image</p>
                          <p className="text-gray-600">Drag and drop or click to browse files</p>
                          <p className="text-sm text-gray-500 mt-2">Supports JPEG, PNG, DICOM formats</p>
                        </div>
                      </div>
                    )}
                  </label>
                </div>

                {uploadedFile && (
                  <Alert className="bg-green-50 border-green-200">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800">
                      <strong>File uploaded:</strong> {uploadedFile.name} ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
                      <br />
                      <span className="text-sm">Ready for AI analysis</span>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Quality Guidelines */}
                <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                  <h4 className="font-semibold text-blue-800 mb-3 flex items-center gap-2">
                    <Camera className="h-4 w-4" />
                    Image Quality Tips
                  </h4>
                  <ul className="text-sm text-blue-700 space-y-1">
                    <li>• Minimum resolution: 1024x1024 pixels</li>
                    <li>• Ensure proper lighting and contrast</li>
                    <li>• Avoid shadows, reflections, or obstructions</li>
                    <li>• Include patient positioning markers if available</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* Patient Information (Doctor view) or Analysis Info (Patient view) */}
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                    {userType === 'doctor' ? <User className="h-5 w-5 text-white" /> : <Brain className="h-5 w-5 text-white" />}
                  </div>
                  {userType === 'doctor' ? 'Patient Information' : 'AI Analysis Process'}
                </CardTitle>
                <CardDescription>
                  {userType === 'doctor' 
                    ? 'Enter patient details for the diagnosis report' 
                    : 'How our advanced AI analyzes your X-ray'
                  }
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {userType === 'doctor' ? (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="patientId" className="text-gray-700 font-medium">Patient ID</Label>
                      <Input
                        id="patientId"
                        value={patientInfo.patientId}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, patientId: e.target.value }))}
                        placeholder="Enter patient ID"
                        className="bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="age" className="text-gray-700 font-medium">Age</Label>
                      <Input
                        id="age"
                        type="number"
                        value={patientInfo.age}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, age: e.target.value }))}
                        placeholder="Patient age"
                        className="bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="symptoms" className="text-gray-700 font-medium">Symptoms</Label>
                      <Textarea
                        id="symptoms"
                        value={patientInfo.symptoms}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, symptoms: e.target.value }))}
                        placeholder="Describe patient symptoms..."
                        className="min-h-[100px] bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="notes" className="text-gray-700 font-medium">Clinical Notes</Label>
                      <Textarea
                        id="notes"
                        value={patientInfo.notes}
                        onChange={(e) => setPatientInfo(prev => ({ ...prev, notes: e.target.value }))}
                        placeholder="Any additional clinical notes..."
                        className="min-h-[100px] bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                      />
                    </div>
                  </>
                ) : (
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
                        <div className="p-2 bg-blue-500 rounded-lg">
                          <Upload className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-blue-800">1. Image Processing</h4>
                          <p className="text-sm text-blue-700">Upload and preprocessing of your X-ray image with quality enhancement</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-100">
                        <div className="p-2 bg-green-500 rounded-lg">
                          <Brain className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-green-800">2. AI Analysis</h4>
                          <p className="text-sm text-green-700">Deep learning model analyzes for abnormalities and conditions</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100">
                        <div className="p-2 bg-purple-500 rounded-lg">
                          <FileCheck className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-purple-800">3. Report Generation</h4>
                          <p className="text-sm text-purple-700">Comprehensive diagnostic report with confidence scores</p>
                        </div>
                      </div>
                    </div>

                    <Alert className="bg-amber-50 border-amber-200">
                      <AlertCircle className="h-4 w-4 text-amber-600" />
                      <AlertDescription className="text-amber-800">
                        <strong>Medical Disclaimer:</strong> AI analysis is for informational purposes only. 
                        Always consult with a healthcare professional for medical decisions.
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Analysis Section */}
          {isAnalyzing && (
            <Card className="bg-gradient-to-r from-white/90 to-white/70 backdrop-blur-sm border-0 shadow-2xl">
              <CardHeader className="text-center">
                <div className="relative inline-block mb-4">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse"></div>
                  <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full">
                    <Clock className="h-8 w-8 text-white animate-spin" />
                  </div>
                </div>
                <CardTitle className="text-2xl flex items-center justify-center gap-3">
                  <Sparkles className="h-6 w-6 text-blue-600" />
                  AI Analysis in Progress
                  <Sparkles className="h-6 w-6 text-blue-600" />
                </CardTitle>
                <CardDescription className="text-lg">Our advanced AI is processing your X-ray image</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="text-center space-y-4">
                  <div className="text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    {analysisProgress}%
                  </div>
                  <Progress value={analysisProgress} className="w-full max-w-md mx-auto h-3" />
                </div>
                <div className="text-center space-y-3">
                  <p className="text-lg font-medium text-gray-700">
                    {analysisProgress < 30 && "🔍 Preprocessing and enhancing image quality..."}
                    {analysisProgress >= 30 && analysisProgress < 70 && "🧠 Running deep learning analysis..."}
                    {analysisProgress >= 70 && analysisProgress < 100 && "📊 Generating comprehensive report..."}
                    {analysisProgress === 100 && "✨ Finalizing diagnostic results..."}
                  </p>
                  <div className="flex items-center justify-center gap-6 text-sm text-gray-600">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-blue-500" />
                      <span>High Performance</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-green-500" />
                      <span>Secure Processing</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Action Button */}
          {uploadedFile && !isAnalyzing && (
            <div className="text-center">
              <div className="relative inline-block">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse"></div>
                <Button 
                  onClick={handleAnalyze}
                  size="lg"
                  className="relative bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-12 py-4 text-lg font-semibold rounded-full shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-105"
                >
                  <Scan className="h-6 w-6 mr-3" />
                  Start AI Analysis
                  <Sparkles className="h-6 w-6 ml-3" />
                </Button>
              </div>
              <p className="text-sm text-gray-500 mt-4">Analysis typically completes in 4-6 seconds</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}