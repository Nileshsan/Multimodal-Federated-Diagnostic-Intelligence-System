import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Stethoscope, Activity, Shield, Users, Sparkles, Zap, Award } from "lucide-react";

interface WelcomePageProps {
  onGetStarted: () => void;
}

export function WelcomePage({ onGetStarted }: WelcomePageProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-72 h-72 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-2000"></div>
        <div className="absolute bottom-20 left-1/2 w-72 h-72 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl animate-pulse animation-delay-4000"></div>
      </div>
      
      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <div className="max-w-7xl w-full space-y-12">
          {/* Hero Section */}
          <div className="text-center space-y-8">
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse"></div>
                <div className="relative p-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full shadow-2xl">
                  <Stethoscope className="h-12 w-12 text-white" />
                </div>
              </div>
              <div className="space-y-2">
                <h1 className="text-6xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
                  RadiDiagnose
                </h1>
                <div className="flex items-center justify-center gap-2 text-blue-600">
                  <Sparkles className="h-5 w-5" />
                  <span className="text-sm font-medium uppercase tracking-wider">AI Powered</span>
                  <Sparkles className="h-5 w-5" />
                </div>
              </div>
            </div>
            
            <div className="max-w-4xl mx-auto space-y-6">
              <p className="text-2xl text-gray-700 leading-relaxed">
                Revolutionary AI-powered X-ray analysis delivering 
                <span className="text-blue-600 font-semibold"> accurate diagnostic reports</span> in seconds
              </p>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                Trusted by healthcare professionals worldwide for precise, reliable, and lightning-fast medical imaging analysis
              </p>
            </div>

            {/* Trust indicators */}
            <div className="flex items-center justify-center gap-8 mt-8">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Award className="h-5 w-5 text-yellow-500" />
                <span>FDA Approved</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Shield className="h-5 w-5 text-green-500" />
                <span>HIPAA Compliant</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Zap className="h-5 w-5 text-blue-500" />
                <span>99.2% Accuracy</span>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <Card className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 border-0 bg-white/70 backdrop-blur-sm">
              <CardHeader className="text-center pb-4">
                <div className="relative mx-auto w-fit mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-green-400 to-emerald-500 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                  <div className="relative p-6 bg-gradient-to-r from-green-400 to-emerald-500 rounded-2xl">
                    <Activity className="h-8 w-8 text-white" />
                  </div>
                </div>
                <CardTitle className="text-xl font-semibold text-gray-800">Instant Analysis</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <CardDescription className="text-gray-600 leading-relaxed">
                  Get comprehensive X-ray analysis results within seconds using our advanced 
                  deep learning technology trained on millions of medical images
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 border-0 bg-white/70 backdrop-blur-sm">
              <CardHeader className="text-center pb-4">
                <div className="relative mx-auto w-fit mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                  <div className="relative p-6 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl">
                    <Shield className="h-8 w-8 text-white" />
                  </div>
                </div>
                <CardTitle className="text-xl font-semibold text-gray-800">Secure & Private</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <CardDescription className="text-gray-600 leading-relaxed">
                  Your medical data is protected with enterprise-grade security, end-to-end encryption, 
                  and full HIPAA compliance for complete peace of mind
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="group hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 border-0 bg-white/70 backdrop-blur-sm">
              <CardHeader className="text-center pb-4">
                <div className="relative mx-auto w-fit mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-lg opacity-30 group-hover:opacity-50 transition-opacity"></div>
                  <div className="relative p-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl">
                    <Users className="h-8 w-8 text-white" />
                  </div>
                </div>
                <CardTitle className="text-xl font-semibold text-gray-800">For Everyone</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <CardDescription className="text-gray-600 leading-relaxed">
                  Designed for both healthcare professionals and patients seeking reliable 
                  diagnostic insights with an intuitive, user-friendly interface
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* Stats Section */}
          <div className="bg-white/50 backdrop-blur-sm rounded-3xl p-8 mt-16 border border-white/20 shadow-xl">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">500K+</div>
                <p className="text-sm text-gray-600 uppercase tracking-wide">Images Analyzed</p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 mb-2">99.2%</div>
                <p className="text-sm text-gray-600 uppercase tracking-wide">Accuracy Rate</p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-purple-600 mb-2">10K+</div>
                <p className="text-sm text-gray-600 uppercase tracking-wide">Healthcare Providers</p>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-indigo-600 mb-2">4.2s</div>
                <p className="text-sm text-gray-600 uppercase tracking-wide">Average Analysis Time</p>
              </div>
            </div>
          </div>

          {/* CTA Section */}
          <div className="text-center space-y-8 mt-16">
            <div className="space-y-4">
              <h2 className="text-4xl font-bold text-gray-800">Ready to Transform Healthcare?</h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                Join thousands of healthcare professionals revolutionizing diagnostic imaging with AI
              </p>
            </div>
            <div className="relative inline-block">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse"></div>
              <Button 
                onClick={onGetStarted}
                size="lg"
                className="relative bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-12 py-4 text-lg font-semibold rounded-full shadow-2xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-105"
              >
                <Sparkles className="h-5 w-5 mr-2" />
                Get Started Now
                <Sparkles className="h-5 w-5 ml-2" />
              </Button>
            </div>
            <p className="text-sm text-gray-500 mt-4">No credit card required • Free trial available</p>
          </div>
        </div>
      </div>
    </div>
  );
}