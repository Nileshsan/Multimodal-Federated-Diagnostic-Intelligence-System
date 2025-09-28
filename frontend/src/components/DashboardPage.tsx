import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { 
  Upload, 
  History, 
  FileText, 
  TrendingUp, 
  Users, 
  Brain,
  Activity,
  Calendar,
  Settings,
  LogOut,
  Sparkles,
  Zap,
  Target,
  Clock,
  Award
} from "lucide-react";

interface DashboardPageProps {
  userType: 'patient' | 'doctor';
  onNavigate: (page: 'upload' | 'history' | 'profile') => void;
  onLogout: () => void;
}

export function DashboardPage({ userType, onNavigate, onLogout }: DashboardPageProps) {
  // Mock statistics - different for patients vs doctors
  const patientStats = {
    totalAnalyses: 3,
    normalResults: 3,
    pendingReviews: 0,
    avgConfidence: 92
  };

  const doctorStats = {
    totalPatients: 47,
    todayAnalyses: 8,
    pendingReviews: 3,
    avgConfidence: 91
  };

  const stats = userType === 'patient' ? patientStats : doctorStats;

  // Recent activity data
  const recentActivity = userType === 'doctor' ? [
    { id: 1, patient: 'John Smith', finding: 'Normal chest X-ray', time: '2 hours ago', status: 'completed', confidence: 94 },
    { id: 2, patient: 'Sarah Johnson', finding: 'Mild pneumonia', time: '4 hours ago', status: 'reviewed', confidence: 87 },
    { id: 3, patient: 'Michael Brown', finding: 'Fractured rib', time: '6 hours ago', status: 'completed', confidence: 91 },
  ] : [
    { id: 1, finding: 'Normal chest X-ray', time: '2 days ago', status: 'completed', confidence: 94 },
    { id: 2, finding: 'Clear cardiac silhouette', time: '1 week ago', status: 'completed', confidence: 89 },
    { id: 3, finding: 'Normal bone density', time: '2 weeks ago', status: 'completed', confidence: 91 },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-emerald-100 text-emerald-700 border border-emerald-200';
      case 'pending': return 'bg-amber-100 text-amber-700 border border-amber-200';
      case 'reviewed': return 'bg-blue-100 text-blue-700 border border-blue-200';
      default: return 'bg-gray-100 text-gray-700 border border-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-4000"></div>
      </div>
      
      <div className="relative z-10 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-lg opacity-20 animate-pulse"></div>
                <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-xl">
                  <Brain className="h-8 w-8 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
                  {userType === 'doctor' ? 'Doctor Dashboard' : 'Patient Dashboard'}
                </h1>
                <p className="text-gray-600 text-lg mt-1 flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Welcome back to RadiDiagnose AI
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button 
                variant="outline" 
                onClick={() => onNavigate('profile')} 
                className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 hover:bg-white/90 rounded-xl"
              >
                <Settings className="h-4 w-4" />
                Settings
              </Button>
              <Button 
                variant="outline" 
                onClick={onLogout} 
                className="flex items-center gap-2 bg-white/70 backdrop-blur-sm border-white/20 hover:bg-white/90 rounded-xl"
              >
                <LogOut className="h-4 w-4" />
                Logout
              </Button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card 
              className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 cursor-pointer border-0 bg-white/70 backdrop-blur-sm" 
              onClick={() => onNavigate('upload')}
            >
              <CardHeader className="text-center pb-4">
                <div className="relative mx-auto w-fit mb-4">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                  <div className="relative p-6 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl shadow-xl">
                    <Upload className="h-10 w-10 text-white" />
                  </div>
                </div>
                <CardTitle className="text-xl font-semibold text-gray-800">New X-ray Analysis</CardTitle>
                <CardDescription className="text-gray-600">
                  Upload and analyze X-ray images with our advanced AI technology
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <Button className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200">
                  <Sparkles className="h-4 w-4 mr-2" />
                  Start Analysis
                </Button>
              </CardContent>
            </Card>

            <Card 
              className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 cursor-pointer border-0 bg-white/70 backdrop-blur-sm" 
              onClick={() => onNavigate('history')}
            >
              <CardHeader className="text-center pb-4">
                <div className="relative mx-auto w-fit mb-4">
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                  <div className="relative p-6 bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl shadow-xl">
                    <History className="h-10 w-10 text-white" />
                  </div>
                </div>
                <CardTitle className="text-xl font-semibold text-gray-800">View History</CardTitle>
                <CardDescription className="text-gray-600">
                  {userType === 'doctor' ? 'Browse patient analyses and detailed reports' : 'Review your previous X-ray results and trends'}
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <Button variant="outline" className="w-full border-emerald-200 text-emerald-700 hover:bg-emerald-50 font-semibold rounded-xl">
                  <History className="h-4 w-4 mr-2" />
                  Browse History
                </Button>
              </CardContent>
            </Card>

            {userType === 'doctor' && (
              <Card className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 border-0 bg-white/70 backdrop-blur-sm">
                <CardHeader className="text-center pb-4">
                  <div className="relative mx-auto w-fit mb-4">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                    <div className="relative p-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl shadow-xl">
                      <Users className="h-10 w-10 text-white" />
                    </div>
                  </div>
                  <CardTitle className="text-xl font-semibold text-gray-800">Patient Management</CardTitle>
                  <CardDescription className="text-gray-600">
                    Manage patient records, assignments, and medical histories
                  </CardDescription>
                </CardHeader>
                <CardContent className="text-center">
                  <Button variant="outline" className="w-full border-purple-200 text-purple-700 hover:bg-purple-50 font-semibold rounded-xl">
                    <Users className="h-4 w-4 mr-2" />
                    Manage Patients
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Statistics */}
          <Card className="bg-gradient-to-r from-white/80 to-white/60 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-2xl">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                {userType === 'doctor' ? 'Practice Statistics' : 'Your Statistics'}
              </CardTitle>
              <CardDescription className="text-lg">
                {userType === 'doctor' ? 'Overview of your diagnostic activity and performance' : 'Your X-ray analysis summary and insights'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl border border-blue-200">
                  <div className="text-4xl font-bold text-blue-600 mb-2">
                    {userType === 'doctor' ? stats.totalPatients : stats.totalAnalyses}
                  </div>
                  <p className="text-sm text-blue-700 font-medium uppercase tracking-wide">
                    {userType === 'doctor' ? 'Total Patients' : 'Total Analyses'}
                  </p>
                </div>
                <div className="text-center p-6 bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-2xl border border-emerald-200">
                  <div className="text-4xl font-bold text-emerald-600 mb-2">
                    {userType === 'doctor' ? stats.todayAnalyses : stats.normalResults}
                  </div>
                  <p className="text-sm text-emerald-700 font-medium uppercase tracking-wide">
                    {userType === 'doctor' ? "Today's Analyses" : 'Normal Results'}
                  </p>
                </div>
                <div className="text-center p-6 bg-gradient-to-br from-amber-50 to-amber-100 rounded-2xl border border-amber-200">
                  <div className="text-4xl font-bold text-amber-600 mb-2">
                    {stats.pendingReviews}
                  </div>
                  <p className="text-sm text-amber-700 font-medium uppercase tracking-wide">Pending Reviews</p>
                </div>
                <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl border border-purple-200">
                  <div className="text-4xl font-bold text-purple-600 mb-2">
                    {stats.avgConfidence}%
                  </div>
                  <p className="text-sm text-purple-700 font-medium uppercase tracking-wide">Avg. Confidence</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Recent Activity */}
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-gradient-to-r from-emerald-500 to-green-500 rounded-lg">
                    <Activity className="h-5 w-5 text-white" />
                  </div>
                  Recent Activity
                </CardTitle>
                <CardDescription>
                  {userType === 'doctor' ? 'Latest patient analyses and diagnostic results' : 'Your recent X-ray results and findings'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-white rounded-xl border border-gray-100 hover:shadow-md transition-shadow">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          {userType === 'doctor' && (
                            <span className="font-semibold text-gray-800">{activity.patient}</span>
                          )}
                          <Badge className={getStatusColor(activity.status)}>
                            {activity.status}
                          </Badge>
                        </div>
                        <p className="text-gray-700 font-medium">{activity.finding}</p>
                        <div className="flex items-center gap-4 mt-2">
                          <p className="text-sm text-gray-500 flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {activity.time}
                          </p>
                          <div className="flex items-center gap-1 text-sm">
                            <Target className="h-3 w-3 text-blue-500" />
                            <span className="text-blue-600 font-medium">{activity.confidence}% confidence</span>
                          </div>
                        </div>
                      </div>
                      <Button variant="ghost" size="sm" className="ml-4">
                        <FileText className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* AI Performance */}
            <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
                    <Brain className="h-5 w-5 text-white" />
                  </div>
                  AI Performance Metrics
                </CardTitle>
                <CardDescription>System reliability and accuracy indicators</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Diagnostic Accuracy</span>
                    <div className="flex items-center gap-2">
                      <Award className="h-4 w-4 text-emerald-500" />
                      <span className="font-bold text-emerald-600">99.2%</span>
                    </div>
                  </div>
                  <Progress value={99.2} className="h-3 bg-gray-100" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">System Uptime</span>
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-blue-500" />
                      <span className="font-bold text-blue-600">99.9%</span>
                    </div>
                  </div>
                  <Progress value={99.9} className="h-3 bg-gray-100" />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-700">Processing Speed</span>
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-purple-500" />
                      <span className="font-bold text-purple-600">4.2s avg</span>
                    </div>
                  </div>
                  <Progress value={85} className="h-3 bg-gray-100" />
                </div>
                <div className="pt-4 border-t bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-100">
                  <p className="text-sm text-gray-600 flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    Last updated: {new Date().toLocaleDateString()}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Tips & Best Practices */}
          <Card className="bg-gradient-to-r from-white/80 to-white/60 backdrop-blur-sm border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-xl">
                <div className="p-2 bg-gradient-to-r from-amber-500 to-orange-500 rounded-lg">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                Tips & Best Practices
              </CardTitle>
              <CardDescription>Expert recommendations for optimal results</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <h4 className="font-semibold text-gray-800 flex items-center gap-2">
                    <Upload className="h-4 w-4 text-blue-500" />
                    Image Quality Tips:
                  </h4>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      Ensure X-rays are high resolution (minimum 1024x1024 pixels)
                    </li>
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      Use proper lighting when photographing film X-rays
                    </li>
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      Avoid shadows, reflections, and image distortion
                    </li>
                  </ul>
                </div>
                <div className="space-y-4">
                  <h4 className="font-semibold text-gray-800 flex items-center gap-2">
                    <Brain className="h-4 w-4 text-purple-500" />
                    Diagnostic Guidelines:
                  </h4>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                      AI analysis supplements, not replaces, clinical judgment
                    </li>
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                      Always correlate findings with patient symptoms
                    </li>
                    <li className="flex items-start gap-2 text-gray-600">
                      <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                      Consider follow-up imaging for borderline cases
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}