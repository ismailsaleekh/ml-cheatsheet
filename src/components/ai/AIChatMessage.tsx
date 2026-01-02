/**
 * AIChatMessage - Individual chat message component
 */
import React from 'react';
import { User, Bot, ExternalLink } from 'lucide-react';
import { cn } from '@/utils/cn';
import type { ChatMessage } from '@/types/rag';

interface AIChatMessageProps {
  message: ChatMessage;
  onSourceClick?: (conceptId: string) => void;
  className?: string;
}

export const AIChatMessage: React.FC<AIChatMessageProps> = ({
  message,
  onSourceClick,
  className,
}) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return null; // Don't display system messages
  }

  return (
    <div
      className={cn(
        'flex gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row',
        className
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser
            ? 'bg-blue-100 dark:bg-blue-900/30'
            : 'bg-purple-100 dark:bg-purple-900/30'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-blue-600 dark:text-blue-400" />
        ) : (
          <Bot className="w-4 h-4 text-purple-600 dark:text-purple-400" />
        )}
      </div>

      {/* Message content */}
      <div
        className={cn(
          'flex-1 max-w-[80%]',
          isUser ? 'text-right' : 'text-left'
        )}
      >
        <div
          className={cn(
            'inline-block px-4 py-2 rounded-2xl',
            isUser
              ? 'bg-blue-500 text-white rounded-br-md'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-bl-md'
          )}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1 justify-start">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              Sources:
            </span>
            {message.sources.map((sourceId) => (
              <button
                key={sourceId}
                onClick={() => onSourceClick?.(sourceId)}
                className={cn(
                  'inline-flex items-center gap-1 px-2 py-0.5 rounded-full',
                  'text-xs bg-gray-100 dark:bg-gray-800',
                  'text-purple-600 dark:text-purple-400',
                  'hover:bg-gray-200 dark:hover:bg-gray-700',
                  'transition-colors'
                )}
              >
                <span>{sourceId}</span>
                <ExternalLink className="w-3 h-3" />
              </button>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <div
          className={cn(
            'mt-1 text-xs text-gray-400 dark:text-gray-500',
            isUser ? 'text-right' : 'text-left'
          )}
        >
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </div>
      </div>
    </div>
  );
};

export default AIChatMessage;
