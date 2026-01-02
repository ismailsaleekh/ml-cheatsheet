/**
 * AIChatInterface - Full chat interface for AI Q&A
 */
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageCircle, Sparkles } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useAppDispatch } from '@/context/AppContext';
import type { ChatMessage } from '@/types/rag';
import { AIChatMessage } from './AIChatMessage';
import { AIChatInput } from './AIChatInput';

interface AIChatInterfaceProps {
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

const SUGGESTIONS = [
  'What is gradient descent?',
  'Explain overfitting',
  'How does regularization work?',
  'What is the bias-variance tradeoff?',
];

export const AIChatInterface: React.FC<AIChatInterfaceProps> = ({
  isOpen,
  onClose,
  className,
}) => {
  const dispatch = useAppDispatch();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (content: string) => {
    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Simulate AI response (placeholder - would integrate with OpenAI)
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: `I understand you're asking about "${content}". AI chat with full responses requires an OpenAI API key configuration. In the meantime, you can use the semantic search feature to find relevant concepts, or browse the cheatsheet directly.`,
        timestamp: new Date().toISOString(),
        sources: [],
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1000);
  };

  const handleSourceClick = (conceptId: string) => {
    dispatch({ type: 'EXPAND_TO', payload: conceptId });
    dispatch({ type: 'SELECT_CONCEPT', payload: conceptId });
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className={cn(
            'fixed bottom-4 right-4 z-50',
            'w-full max-w-md h-[600px] max-h-[80vh]',
            'bg-white dark:bg-gray-900',
            'rounded-2xl shadow-2xl',
            'border border-gray-200 dark:border-gray-700',
            'flex flex-col overflow-hidden',
            className
          )}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-purple-500 to-blue-500">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-white/20 rounded-lg">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">ML Assistant</h3>
                <p className="text-xs text-white/80">Ask about concepts</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={onClose}
                className="p-2 rounded-lg text-white/80 hover:text-white hover:bg-white/20 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-4">
                <div className="p-4 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-4">
                  <MessageCircle className="w-8 h-8 text-purple-500" />
                </div>
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Ask me anything about ML
                </h4>
                <p className="text-sm text-gray-500 dark:text-gray-400 max-w-xs">
                  I can help explain concepts, compare techniques, or answer your machine learning questions.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <AIChatMessage
                  key={message.id}
                  message={message}
                  onSourceClick={handleSourceClick}
                />
              ))
            )}
            {isLoading && (
              <div className="flex items-center gap-2 text-purple-500">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-sm">Thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <AIChatInput
              onSend={handleSend}
              isLoading={isLoading}
              suggestions={messages.length === 0 ? SUGGESTIONS : []}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AIChatInterface;
