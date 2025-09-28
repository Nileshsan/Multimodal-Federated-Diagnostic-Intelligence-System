import { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { UserCheck, Stethoscope, ArrowLeft, Sparkles, Lock, Mail } from "lucide-react";

interface LoginPageProps {
  onLogin: (userType: 'patient' | 'doctor', userData: any) => void;
  onBack: () => void;
}

export function LoginPage({ onLogin, onBack }: LoginPageProps) {
  const [activeTab, setActiveTab] = useState<'patient' | 'doctor'>('patient');
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLogin(activeTab, formData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/4 left-10 w-64 h-64 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-10 w-64 h-64 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
      </div>
      
      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <div className="max-w-md w-full space-y-8">
          {/* Back Button */}
          <Button 
            variant="ghost" 
            onClick={onBack}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-800 hover:bg-white/50 backdrop-blur-sm rounded-full px-4 py-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Welcome
          </Button>

          {/* Header */}
          <div className="text-center space-y-4">
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-lg opacity-20 animate-pulse"></div>
              <div className="relative p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full">
                <Lock className="h-8 w-8 text-white" />
              </div>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Welcome Back
            </h1>
            <p className="text-gray-600 text-lg">Access your RadiDiagnose account</p>
          </div>

          {/* Login Form */}
          <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-2xl">
            <CardHeader className="text-center pb-6">
              <CardTitle className="text-2xl font-semibold text-gray-800">Sign In</CardTitle>
              <CardDescription className="text-gray-600">
                Select your role and enter your credentials
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'patient' | 'doctor')}>
                <TabsList className="grid w-full grid-cols-2 bg-gray-100/50 p-1 rounded-xl">
                  <TabsTrigger 
                    value="patient" 
                    className="flex items-center gap-2 rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
                  >
                    <UserCheck className="h-4 w-4" />
                    Patient
                  </TabsTrigger>
                  <TabsTrigger 
                    value="doctor" 
                    className="flex items-center gap-2 rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
                  >
                    <Stethoscope className="h-4 w-4" />
                    Doctor
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="patient" className="space-y-6 mt-8">
                  <div className="text-center p-4 bg-blue-50 rounded-xl border border-blue-100">
                    <UserCheck className="h-12 w-12 text-blue-600 mx-auto mb-2" />
                    <h3 className="font-medium text-blue-800">Patient Portal</h3>
                    <p className="text-sm text-blue-600 mt-1">
                      Access your medical records and analysis history
                    </p>
                  </div>
                  
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="patient-email" className="text-gray-700 font-medium">Email Address</Label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                        <Input
                          id="patient-email"
                          name="email"
                          type="email"
                          placeholder="patient@example.com"
                          value={formData.email}
                          onChange={handleInputChange}
                          className="pl-11 h-12 bg-white/50 border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-xl"
                          required
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="patient-password" className="text-gray-700 font-medium">Password</Label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                        <Input
                          id="patient-password"
                          name="password"
                          type="password"
                          value={formData.password}
                          onChange={handleInputChange}
                          className="pl-11 h-12 bg-white/50 border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-xl"
                          required
                        />
                      </div>
                    </div>
                    <Button 
                      type="submit" 
                      className="w-full h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-[1.02]"
                    >
                      <Sparkles className="h-4 w-4 mr-2" />
                      Sign In as Patient
                    </Button>
                  </form>
                </TabsContent>

                <TabsContent value="doctor" className="space-y-6 mt-8">
                  <div className="text-center p-4 bg-purple-50 rounded-xl border border-purple-100">
                    <Stethoscope className="h-12 w-12 text-purple-600 mx-auto mb-2" />
                    <h3 className="font-medium text-purple-800">Healthcare Professional</h3>
                    <p className="text-sm text-purple-600 mt-1">
                      Access professional diagnostic tools and patient management
                    </p>
                  </div>
                  
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="doctor-email" className="text-gray-700 font-medium">Professional Email</Label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                        <Input
                          id="doctor-email"
                          name="email"
                          type="email"
                          placeholder="doctor@hospital.com"
                          value={formData.email}
                          onChange={handleInputChange}
                          className="pl-11 h-12 bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                          required
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="doctor-password" className="text-gray-700 font-medium">Password</Label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                        <Input
                          id="doctor-password"
                          name="password"
                          type="password"
                          value={formData.password}
                          onChange={handleInputChange}
                          className="pl-11 h-12 bg-white/50 border-gray-200 focus:border-purple-400 focus:ring-purple-400 rounded-xl"
                          required
                        />
                      </div>
                    </div>
                    <Button 
                      type="submit" 
                      className="w-full h-12 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-[1.02]"
                    >
                      <Sparkles className="h-4 w-4 mr-2" />
                      Sign In as Doctor
                    </Button>
                  </form>
                </TabsContent>
              </Tabs>

              <div className="mt-8 text-center space-y-4">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-200"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white text-gray-500">Don't have an account?</span>
                  </div>
                </div>
                <Button 
                  variant="link" 
                  className="text-blue-600 hover:text-blue-800 font-medium"
                >
                  Create new account →
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Security Notice */}
          <div className="text-center text-sm text-gray-500 bg-white/50 backdrop-blur-sm rounded-xl p-4 border border-white/20">
            <Lock className="h-4 w-4 inline mr-2" />
            Your data is protected with enterprise-grade security
          </div>
        </div>
      </div>
    </div>
  );
}