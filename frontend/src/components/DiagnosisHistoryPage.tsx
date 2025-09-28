import { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { generateHistoryReport } from "../utils/reportGenerator";
import { 
  ArrowLeft, 
  Search, 
  Calendar,
  Eye,
  Download,
  Filter,
  FileText,
  TrendingUp,
  Clock,
  Target,
  Users
} from "lucide-react";

interface HistoryItem {
  id: string;
  date: string;
  patientId?: string;
  patientName?: string;
  confidence: number;
  primaryFinding: string;
  severity: string;
  status: 'completed' | 'pending' | 'reviewed';
}

interface DiagnosisHistoryPageProps {
  userType: 'patient' | 'doctor';
  onBack: () => void;
  onViewDetails: (item: HistoryItem) => void;
}

export function DiagnosisHistoryPage({ userType, onBack, onViewDetails }: DiagnosisHistoryPageProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [isDownloading, setIsDownloading] = useState(false);

  // Mock data - different for patients vs doctors
  const mockHistoryData: HistoryItem[] = userType === 'doctor' ? [
    {
      id: '1',
      date: '2024-01-15T10:30:00Z',
      patientId: 'P001',
      patientName: 'John Smith',
      confidence: 94,
      primaryFinding: 'Normal chest X-ray',
      severity: 'Normal',
      status: 'completed'
    },
    {
      id: '2',
      date: '2024-01-14T14:20:00Z',
      patientId: 'P002',
      patientName: 'Sarah Johnson',
      confidence: 87,
      primaryFinding: 'Mild pneumonia',
      severity: 'Mild',
      status: 'reviewed'
    },
    {
      id: '3',
      date: '2024-01-13T09:15:00Z',
      patientId: 'P003',
      patientName: 'Michael Brown',
      confidence: 91,
      primaryFinding: 'Fractured rib',
      severity: 'Moderate',
      status: 'completed'
    },
    {
      id: '4',
      date: '2024-01-12T16:45:00Z',
      patientId: 'P004',
      patientName: 'Emily Davis',
      confidence: 96,
      primaryFinding: 'Clear lung fields',
      severity: 'Normal',
      status: 'completed'
    },
    {
      id: '5',
      date: '2024-01-11T11:30:00Z',
      patientId: 'P005',
      patientName: 'Robert Wilson',
      confidence: 82,
      primaryFinding: 'Possible nodule',
      severity: 'Moderate',
      status: 'pending'
    }
  ] : [
    {
      id: '1',
      date: '2024-01-15T10:30:00Z',
      confidence: 94,
      primaryFinding: 'Normal chest X-ray',
      severity: 'Normal',
      status: 'completed'
    },
    {
      id: '2',
      date: '2024-01-10T14:20:00Z',
      confidence: 89,
      primaryFinding: 'Clear cardiac silhouette',
      severity: 'Normal',
      status: 'completed'
    },
    {
      id: '3',
      date: '2024-01-05T09:15:00Z',
      confidence: 91,
      primaryFinding: 'Normal bone density',
      severity: 'Normal',
      status: 'completed'
    }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'normal': return 'bg-green-100 text-green-800 border border-green-200';
      case 'mild': return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
      case 'moderate': return 'bg-orange-100 text-orange-800 border border-orange-200';
      case 'severe': return 'bg-red-100 text-red-800 border border-red-200';
      default: return 'bg-gray-100 text-gray-800 border border-gray-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 border border-green-200';
      case 'pending': return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
      case 'reviewed': return 'bg-blue-100 text-blue-800 border border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border border-gray-200';
    }
  };

  const filteredHistory = mockHistoryData.filter(item => {
    const matchesSearch = userType === 'doctor' 
      ? (item.patientName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
         item.patientId?.toLowerCase().includes(searchTerm.toLowerCase()) ||
         item.primaryFinding.toLowerCase().includes(searchTerm.toLowerCase()))
      : item.primaryFinding.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = filterStatus === 'all' || item.status === filterStatus;
    const matchesSeverity = filterSeverity === 'all' || item.severity.toLowerCase() === filterSeverity;
    
    return matchesSearch && matchesStatus && matchesSeverity;
  });

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleDownloadHistoryReport = async () => {
    try {
      setIsDownloading(true);
      generateHistoryReport(filteredHistory, userType);
      
      // Show success feedback
      setTimeout(() => {
        setIsDownloading(false);
      }, 2000);
    } catch (error) {
      console.error('Error generating history report:', error);
      setIsDownloading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-15">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
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
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 px-4 py-2 rounded-xl">
                <Clock className="h-4 w-4" />
                {filteredHistory.length} Records
              </Badge>
              <Button 
                variant="outline"
                onClick={handleDownloadHistoryReport}
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
                    Export History
                  </>
                )}
              </Button>
            </div>
          </div>

          <div className="text-center space-y-4">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-20 animate-pulse"></div>
              <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-xl">
                <FileText className="h-12 w-12 text-white" />
              </div>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Diagnosis History
            </h1>
            <p className="text-gray-600 text-lg">
              {userType === 'doctor' ? 'View and manage patient X-ray analyses' : 'Your previous X-ray analysis results'}
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">{mockHistoryData.length}</div>
                <p className="text-gray-600 font-medium">Total Analyses</p>
              </CardContent>
            </Card>
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">
                  {mockHistoryData.filter(item => item.severity === 'Normal').length}
                </div>
                <p className="text-gray-600 font-medium">Normal Results</p>
              </CardContent>
            </Card>
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-orange-600 mb-2">
                  {mockHistoryData.filter(item => item.status === 'pending').length}
                </div>
                <p className="text-gray-600 font-medium">Pending Review</p>
              </CardContent>
            </Card>
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">
                  {Math.round(mockHistoryData.reduce((acc, item) => acc + item.confidence, 0) / mockHistoryData.length)}%
                </div>
                <p className="text-gray-600 font-medium">Avg. Confidence</p>
              </CardContent>
            </Card>
          </div>

          {/* Filters */}
          <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
                  <Filter className="h-5 w-5 text-white" />
                </div>
                Filters & Search
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder={userType === 'doctor' ? "Search patients or findings..." : "Search findings..."}
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-white/50 border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-xl"
                  />
                </div>
                <Select value={filterStatus} onValueChange={setFilterStatus}>
                  <SelectTrigger className="bg-white/50 border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-xl">
                    <SelectValue placeholder="Filter by status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="reviewed">Reviewed</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={filterSeverity} onValueChange={setFilterSeverity}>
                  <SelectTrigger className="bg-white/50 border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-xl">
                    <SelectValue placeholder="Filter by severity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severity</SelectItem>
                    <SelectItem value="normal">Normal</SelectItem>
                    <SelectItem value="mild">Mild</SelectItem>
                    <SelectItem value="moderate">Moderate</SelectItem>
                    <SelectItem value="severe">Severe</SelectItem>
                  </SelectContent>
                </Select>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setSearchTerm('');
                    setFilterStatus('all');
                    setFilterSeverity('all');
                  }}
                  className="bg-white/50 border-gray-200 hover:bg-white/90 rounded-xl"
                >
                  Clear Filters
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* History List */}
          <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-xl">
                <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg">
                  <FileText className="h-5 w-5 text-white" />
                </div>
                Analysis History
              </CardTitle>
              <CardDescription className="text-lg">
                {filteredHistory.length} of {mockHistoryData.length} records shown
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredHistory.map((item) => (
                  <div key={item.id} className="bg-gradient-to-r from-gray-50 to-white p-6 rounded-xl border border-gray-100 hover:shadow-md transition-all duration-200">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <h4 className="font-semibold text-gray-800 text-lg">{item.primaryFinding}</h4>
                          <Badge className={getSeverityColor(item.severity)}>
                            {item.severity}
                          </Badge>
                          <Badge className={getStatusColor(item.status)}>
                            {item.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-6 text-gray-600">
                          <span className="flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            {formatDate(item.date)}
                          </span>
                          {userType === 'doctor' && item.patientName && (
                            <span className="flex items-center gap-2">
                              <Users className="h-4 w-4" />
                              {item.patientName} ({item.patientId})
                            </span>
                          )}
                          <span className="flex items-center gap-2">
                            <Target className="h-4 w-4 text-blue-500" />
                            <span className="text-blue-600 font-medium">{item.confidence}% confidence</span>
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onViewDetails(item)}
                          className="flex items-center gap-2 bg-white/70 hover:bg-white/90 rounded-xl"
                        >
                          <Eye className="h-4 w-4" />
                          View Details
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => console.log('Download individual report', item.id)}
                          className="flex items-center gap-2 hover:bg-white/50 rounded-xl"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
                
                {filteredHistory.length === 0 && (
                  <div className="text-center py-16">
                    <div className="relative inline-block mb-6">
                      <div className="absolute inset-0 bg-gray-200 rounded-full blur-lg opacity-30"></div>
                      <div className="relative p-6 bg-gray-100 rounded-full">
                        <FileText className="h-12 w-12 text-gray-400" />
                      </div>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">No records found</h3>
                    <p className="text-gray-600 max-w-md mx-auto">
                      {searchTerm || filterStatus !== 'all' || filterSeverity !== 'all'
                        ? 'Try adjusting your filters or search terms to find what you\'re looking for'
                        : 'No diagnosis history available yet. Start by analyzing your first X-ray image'
                      }
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}