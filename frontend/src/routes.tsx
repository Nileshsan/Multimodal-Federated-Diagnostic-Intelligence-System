import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import App from './App';

const router = createBrowserRouter(
  [
    {
      path: '/Multimodal-Federated-Diagnostic-Intelligence-System/*',
      element: <App />,
    },
  ],
  {
    basename: '/Multimodal-Federated-Diagnostic-Intelligence-System',
  }
);

export function Routes() {
  return <RouterProvider router={router} />;
}