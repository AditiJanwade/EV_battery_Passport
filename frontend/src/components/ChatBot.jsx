
import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Bot } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const ChatBot = ({ healthData }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm your EV Health Assistant. Ask me about your battery report, degradation, or maintenance tips.", sender: 'bot' }
    ]);
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), text: input, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        const currentInput = input;
        setInput('');

        // Show typing indicator (optional, simplified here as instant response)
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: currentInput,
                    context: healthData // Pass the full analysis result
                })
            });

            const data = await response.json();
            const botResponse = { id: Date.now() + 1, text: data.response, sender: 'bot' };
            setMessages(prev => [...prev, botResponse]);
        } catch (error) {
            console.error("Chat Error:", error);
            const errorResponse = { id: Date.now() + 1, text: "I'm having trouble connecting to the server. Please ensure the backend is running.", sender: 'bot' };
            setMessages(prev => [...prev, errorResponse]);
        }
    };

    return (
        <>
            <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setIsOpen(true)}
                className={`fixed bottom-6 right-6 p-4 rounded-full shadow-lg z-50 transition-colors ${isOpen ? 'hidden' : 'bg-brand-blue text-white'}`}
            >
                <MessageSquare className="w-6 h-6" />
            </motion.button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 50, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 50, scale: 0.9 }}
                        className="fixed bottom-6 right-6 w-80 md:w-96 h-[500px] bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-4 bg-slate-900 border-b border-slate-700 flex justify-between items-center">
                            <div className="flex items-center space-x-2">
                                <div className="p-2 bg-brand-blue/20 rounded-lg">
                                    <Bot className="w-5 h-5 text-brand-blue" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">AI Assistant</h3>
                                    <p className="text-xs text-green-400 flex items-center">
                                        <span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span> Online
                                    </p>
                                </div>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-800/50">
                            {messages.map((msg) => (
                                <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] p-3 rounded-2xl text-sm ${msg.sender === 'user'
                                        ? 'bg-brand-blue text-white rounded-tr-none'
                                        : 'bg-slate-700 text-slate-200 rounded-tl-none'
                                        }`}>
                                        {msg.text}
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input */}
                        <form onSubmit={handleSend} className="p-4 bg-slate-900 border-t border-slate-700">
                            <div className="flex items-center space-x-2">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Ask a question..."
                                    className="flex-1 bg-slate-800 text-white rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand-blue"
                                />
                                <button type="submit" className="p-2 bg-brand-blue text-white rounded-lg hover:bg-blue-600 transition-colors">
                                    <Send className="w-4 h-4" />
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
