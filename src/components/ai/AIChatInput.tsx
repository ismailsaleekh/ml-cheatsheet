/**
 * AIChatInput - Chat input with suggestions
 */
import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';

interface AIChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  suggestions?: string[];
  className?: string;
}

export const AIChatInput: React.FC<AIChatInputProps> = ({
  onSend,
  isLoading = false,
  placeholder = 'Ask about ML concepts...',
  suggestions = [],
  className,
}) => {
  const [message, setMessage] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion);
    inputRef.current?.focus();
  };

  return (
    <div className={cn('space-y-2', className)}>
      {/* Suggestions */}
      {suggestions.length > 0 && !message && (
        <div className="flex flex-wrap gap-2">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              className={cn(
                'px-3 py-1.5 rounded-full text-xs',
                'bg-gray-100 dark:bg-gray-800',
                'text-gray-600 dark:text-gray-400',
                'hover:bg-gray-200 dark:hover:bg-gray-700',
                'transition-colors'
              )}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            ref={inputRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            rows={1}
            className={cn(
              'w-full px-4 py-3 pr-12 rounded-xl resize-none',
              'bg-gray-100 dark:bg-gray-800',
              'border border-transparent',
              'focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20',
              'text-gray-900 dark:text-white',
              'placeholder:text-gray-500 dark:placeholder:text-gray-400',
              'disabled:opacity-50',
              'max-h-30'
            )}
          />
        </div>

        <button
          type="submit"
          disabled={!message.trim() || isLoading}
          className={cn(
            'p-3 rounded-xl transition-all',
            'bg-purple-500 text-white',
            'hover:bg-purple-600',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2'
          )}
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>

      {/* Hint */}
      <p className="text-xs text-gray-400 dark:text-gray-500 text-center">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
};

export default AIChatInput;
