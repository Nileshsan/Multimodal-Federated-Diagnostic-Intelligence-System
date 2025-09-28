
  import { StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { Routes } from "./routes";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./index.css";

const LoadingSpinner = () => (
  <div className="min-h-screen flex items-center justify-center">
    <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-gray-900"></div>
  </div>
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes />
      </Suspense>
    </ErrorBoundary>
  </StrictMode>
);
  