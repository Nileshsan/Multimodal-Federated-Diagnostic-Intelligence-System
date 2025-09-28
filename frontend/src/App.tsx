import { useState } from "react";
import { WelcomePage } from "./components/WelcomePage";
import { LoginPage } from "./components/LoginPage";
import { PersonalDetailsPage } from "./components/PersonalDetailsPage";
import { DashboardPage } from "./components/DashboardPage";
import { XrayUploadPage } from "./components/XrayUploadPage";
import { DiagnosisResultsPage } from "./components/DiagnosisResultsPage";
import { DiagnosisHistoryPage } from "./components/DiagnosisHistoryPage";

type Page = 'welcome' | 'login' | 'dashboard' | 'profile' | 'upload' | 'results' | 'history';
type UserType = 'patient' | 'doctor' | null;

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('welcome');
  const [userType, setUserType] = useState<UserType>(null);
  const [diagnosisResults, setDiagnosisResults] = useState<any>(null);

  const handleGetStarted = () => {
    setCurrentPage('login');
  };

  const handleLogin = (type: 'patient' | 'doctor', userData: any) => {
    setUserType(type);
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setUserType(null);
    setCurrentPage('welcome');
    setDiagnosisResults(null);
  };

  const handleBackToWelcome = () => {
    setCurrentPage('welcome');
  };

  const handleNavigate = (page: 'upload' | 'history' | 'dashboard' | 'profile') => {
    setCurrentPage(page);
  };

  const handleAnalysisComplete = (results: any) => {
    setDiagnosisResults(results);
    setCurrentPage('results');
  };

  const handleBackToDashboard = () => {
    setCurrentPage('dashboard');
  };

  const handleBackToUpload = () => {
    setCurrentPage('upload');
  };

  const handleNewAnalysis = () => {
    setDiagnosisResults(null);
    setCurrentPage('upload');
  };

  const handleViewDetails = (item: any) => {
    // Mock setting results for history item
    setDiagnosisResults({
      confidence: item.confidence,
      findings: [{ condition: item.primaryFinding, severity: item.severity, confidence: item.confidence }],
      recommendations: ["Follow up as needed", "Continue monitoring"],
      technicalQuality: "Good",
      timestamp: item.date
    });
    setCurrentPage('results');
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'welcome':
        return <WelcomePage onGetStarted={handleGetStarted} />;
      case 'login':
        return <LoginPage onLogin={handleLogin} onBack={handleBackToWelcome} />;
      case 'dashboard':
        return userType ? (
          <DashboardPage 
            userType={userType} 
            onNavigate={handleNavigate}
            onLogout={handleLogout} 
          />
        ) : null;
      case 'profile':
        return userType ? (
          <PersonalDetailsPage 
            userType={userType} 
            onLogout={handleLogout}
            onNavigate={handleNavigate}
          />
        ) : null;
      case 'upload':
        return userType ? (
          <XrayUploadPage
            userType={userType}
            onBack={handleBackToDashboard}
            onAnalysisComplete={handleAnalysisComplete}
          />
        ) : null;
      case 'results':
        return userType && diagnosisResults ? (
          <DiagnosisResultsPage
            userType={userType}
            results={diagnosisResults}
            onBack={handleBackToUpload}
            onNewAnalysis={handleNewAnalysis}
          />
        ) : null;
      case 'history':
        return userType ? (
          <DiagnosisHistoryPage
            userType={userType}
            onBack={handleBackToDashboard}
            onViewDetails={handleViewDetails}
          />
        ) : null;
      default:
        return <WelcomePage onGetStarted={handleGetStarted} />;
    }
  };

  return (
    <div className="size-full">
      {renderCurrentPage()}
    </div>
  );
}