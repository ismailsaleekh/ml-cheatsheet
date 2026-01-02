/**
 * Main App component - Root of the application
 */
import { useEffect } from 'react';
import { AppProvider } from '@/context/AppContext';
import { ProgressProvider } from '@/context/ProgressContext';
import { Header } from '@/components/layout/Header';
import { MainContent } from '@/components/layout/MainContent';
import { DetailModal } from '@/components/detail/DetailModal';
import { useRAG } from '@/hooks/useRAG';

/**
 * Inner app component that can use hooks
 */
function AppContent() {
  // Initialize RAG service
  const { initialize } = useRAG();

  useEffect(() => {
    // Initialize RAG on mount (will load embeddings if available)
    initialize();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      <Header />
      <MainContent />

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 py-6">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            ML Cheatsheet - Interactive Machine Learning Reference
          </p>
        </div>
      </footer>

      {/* Detail Modal */}
      <DetailModal />
    </div>
  );
}

function App() {
  return (
    <AppProvider>
      <ProgressProvider>
        <AppContent />
      </ProgressProvider>
    </AppProvider>
  );
}

export default App;
