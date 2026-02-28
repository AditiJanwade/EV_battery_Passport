import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Bot, Zap, TrendingUp, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const ChatBot = ({ healthData }) => {
    const [isOpen, setIsOpen] = useState(false);

    // Dynamic greeting based on context
    const initialMessage = healthData
        ? "Hello! I'm your EV Health Assistant. I see your battery analysis is ready. Ask me about your SOH, Resale Value, or how to improve your range!"
        : "Hello! I'm your EV Health Assistant. I can explain battery terminology or give maintenance tips. \n\n⚠️ For a personalized health report, please run the **Run Analysis** on the dashboard first!";

    const [messages, setMessages] = useState([
        { id: 1, text: initialMessage, sender: 'bot' }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    const handleSend = async (e, forcedText = null) => {
        if (e) e.preventDefault();

        const textToSend = forcedText || input;
        if (!textToSend.trim()) return;

        const userMsg = { id: Date.now(), text: textToSend, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsTyping(true);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: textToSend,
                    context: healthData
                })
            });

            const data = await response.json();

            // Artificial delay for UI smoothness
            setTimeout(() => {
                const botResponse = { id: Date.now() + 1, text: data.response, sender: 'bot' };
                setIsTyping(false);
                setMessages(prev => [...prev, botResponse]);
            }, 600);

        } catch (error) {
            console.error("Chat Error:", error);
            setTimeout(() => {
                setIsTyping(false);
                const errorResponse = { id: Date.now() + 1, text: "I'm having trouble connecting to the server. Please ensure the backend is running.", sender: 'bot' };
                setMessages(prev => [...prev, errorResponse]);
            }, 600);
        }
    };

    const renderText = (text) => {
        // Simple bold markdown parser and line breaks
        return text.split('\n').map((line, i) => {
            const parts = line.split(/(\*\*.*?\*\*)/g);
            return (
                <span key={i}>
                    {parts.map((part, j) => {
                        if (part.startsWith('**') && part.endsWith('**')) {
                            return <strong key={j} className="font-bold text-teal-400">{part.slice(2, -2)}</strong>;
                        }
                        return part;
                    })}
                    <br />
                </span>
            );
        });
    };

    return (
        <>
            <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setIsOpen(true)}
                className={`fixed bottom-6 right-6 p-4 rounded-full shadow-lg z-50 transition-colors ${isOpen ? 'hidden' : 'bg-teal-500 text-white hover:bg-teal-600'}`}
            >
                <MessageSquare className="w-6 h-6" />
            </motion.button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 50, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 50, scale: 0.9 }}
                        className="fixed bottom-6 right-6 w-80 md:w-96 h-[550px] bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-4 bg-slate-900 border-b border-slate-700 flex justify-between items-center">
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-teal-500/20 rounded-lg">
                                    <Bot className="w-5 h-5 text-teal-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white tracking-wide">Battery AI Analyst</h3>
                                    <p className="text-xs text-green-400 flex items-center">
                                        <span className="w-2 h-2 bg-green-500 rounded-full mr-2 shadow-[0_0_8px_rgba(34,197,94,0.8)] animate-pulse"></span> Online
                                    </p>
                                </div>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Quick Replies */}
                        <div className="bg-slate-800/80 p-3 flex gap-2 overflow-x-auto border-b border-slate-700 scrollbar-hide">
                            <button onClick={() => handleSend(null, "What is SOH?")} className="whitespace-nowrap flex items-center text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-full transition-colors">
                                <Info size={12} className="mr-1 text-blue-400" /> What is SOH?
                            </button>
                            <button onClick={() => handleSend(null, "How to improve battery life?")} className="whitespace-nowrap flex items-center text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-full transition-colors">
                                <TrendingUp size={12} className="mr-1 text-green-400" /> Improve Life
                            </button>
                            <button onClick={() => handleSend(null, "What is my resale value?")} className="whitespace-nowrap flex items-center text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-3 py-1.5 rounded-full transition-colors">
                                <Zap size={12} className="mr-1 text-yellow-400" /> Resale Value
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-800/50">
                            {messages.map((msg) => (
                                <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[85%] p-3 text-sm leading-relaxed shadow-sm ${msg.sender === 'user'
                                        ? 'bg-teal-600 text-white rounded-2xl rounded-tr-none'
                                        : 'bg-slate-700 text-slate-200 rounded-2xl rounded-tl-none border border-slate-600'
                                        }`}>
                                        {msg.sender === 'bot' ? renderText(msg.text) : msg.text}
                                    </div>
                                </div>
                            ))}

                            {isTyping && (
                                <div className="flex justify-start">
                                    <div className="bg-slate-700 border border-slate-600 text-slate-400 p-3 rounded-2xl rounded-tl-none flex space-x-1 items-center">
                                        <div className="w-2 h-2 bg-teal-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-2 h-2 bg-teal-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-2 h-2 bg-teal-400 rounded-full animate-bounce"></div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input */}
                        <form onSubmit={(e) => handleSend(e)} className="p-4 bg-slate-900 border-t border-slate-700">
                            <div className="flex items-center space-x-2">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder={healthData ? "Ask about your battery..." : "Ask a general question..."}
                                    className="flex-1 bg-slate-800 text-white border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-teal-500 focus:border-teal-500 transition-all placeholder-slate-500"
                                />
                                <button type="submit" disabled={!input.trim()} className="p-2.5 bg-teal-500 text-white rounded-xl hover:bg-teal-600 disabled:opacity-50 disabled:hover:bg-teal-500 transition-colors shadow-sm">
                                    <Send className="w-4 h-4 ml-0.5" />
                                </button>
                            </div>
                        </form>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default ChatBot;
