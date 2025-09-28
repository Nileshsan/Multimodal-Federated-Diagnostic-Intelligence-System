import { HashRouter, Routes as RouterRoutes, Route } from 'react-router-dom';
import App from './App';

export function Routes() {
  return (
    <HashRouter>
      <RouterRoutes>
        <Route path="/*" element={<App />} />
      </RouterRoutes>
    </HashRouter>
  );
}