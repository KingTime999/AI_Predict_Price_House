import React from 'react';
import { motion } from 'motion/react';
import { HelpCircle, Mail, MessageSquare, Phone, FileText, ExternalLink } from 'lucide-react';

export default function Support() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <header>
        <p className="font-bold text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-2">Architectural Portal</p>
        <h2 className="font-headline text-4xl font-black tracking-tight text-primary">Architectural Support</h2>
        <p className="text-slate-400 font-bold text-sm mt-2">Connect with our consultants and explore our knowledge base.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {[
          { icon: Mail, title: 'Email Support', desc: 'Direct access to our analysts.', action: 'precision@predictmyhome.ai' },
          { icon: MessageSquare, title: 'Live Chat', desc: 'Real-time architectural consultation.', action: 'Start Chat' },
          { icon: Phone, title: 'Priority Line', desc: 'Dedicated line for premium members.', action: '+1 (555) 012-3456' },
        ].map((item) => (
          <div key={item.title} className="bg-white p-10 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.02)] border border-slate-50 hover:shadow-xl transition-all group">
            <div className="w-14 h-14 rounded-2xl bg-slate-50 flex items-center justify-center text-primary mb-8 group-hover:bg-primary group-hover:text-white transition-all">
              <item.icon className="w-6 h-6" />
            </div>
            <h3 className="font-headline text-xl font-bold text-primary mb-2">{item.title}</h3>
            <p className="text-xs font-bold text-slate-400 mb-8 leading-relaxed">{item.desc}</p>
            <button className="text-[10px] font-black uppercase tracking-widest text-primary hover:underline">{item.action}</button>
          </div>
        ))}
      </div>

      <section className="bg-slate-50/50 p-12 rounded-3xl border border-slate-100">
        <h3 className="font-headline text-2xl font-black text-primary mb-10">Knowledge Base</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            'Understanding Predictive Accuracy',
            'Market Trends Analysis Guide',
            'API Integration for Developers',
            'Subscription & Billing Protocols',
            'Data Privacy & Security Standards',
            'Property Valuation Methodology'
          ].map((item) => (
            <button key={item} className="flex items-center justify-between p-6 bg-white rounded-2xl shadow-sm hover:shadow-md transition-all group">
              <div className="flex items-center gap-4">
                <FileText className="w-5 h-5 text-slate-300 group-hover:text-primary transition-colors" />
                <span className="text-sm font-bold text-slate-600 group-hover:text-primary">{item}</span>
              </div>
              <ExternalLink className="w-4 h-4 text-slate-200 group-hover:text-primary" />
            </button>
          ))}
        </div>
      </section>
    </motion.div>
  );
}
