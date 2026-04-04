import React, { useState } from 'react';
import { Mail, Phone, MapPin, Send, Globe, Share2, Network, CheckCircle2 } from 'lucide-react';

export default function Contact() {
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate submission
    setTimeout(() => {
      setLoading(false);
      setSubmitted(true);
    }, 1500);
  };

  return (
    <div className="pt-32 pb-24 px-8 max-w-7xl mx-auto">
      <header className="mb-20 max-w-3xl">
        <div className="inline-block px-3 py-1 bg-secondary/10 text-secondary rounded-full text-xs font-bold tracking-widest uppercase mb-6">Connect with Precision</div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tighter text-primary leading-tight mb-8">Let's architecturalize your real estate future.</h1>
        <p className="text-slate-600 text-lg leading-relaxed font-light">Our team of engineers and market analysts are ready to help you navigate the predictive landscape of property valuation.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
        <section className="lg:col-span-7 bg-white rounded-2xl p-8 md:p-12 shadow-sm border border-slate-100">
          <h2 className="text-2xl font-bold text-primary mb-8 tracking-tight">Professional Inquiry</h2>
          
          {submitted ? (
            <div className="py-12 text-center space-y-4">
              <div className="flex justify-center">
                <CheckCircle2 className="w-16 h-16 text-secondary" />
              </div>
              <h3 className="text-2xl font-bold text-primary">Message Received</h3>
              <p className="text-slate-500 max-w-xs mx-auto">Our architectural consultants will review your inquiry and respond within 24 hours.</p>
              <button 
                onClick={() => setSubmitted(false)}
                className="text-sm font-bold text-secondary hover:underline"
              >
                Send another message
              </button>
            </div>
          ) : (
            <form className="space-y-8" onSubmit={handleSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-2">
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest">Full Name</label>
                  <input className="w-full bg-slate-50 border-none focus:ring-2 focus:ring-primary/20 px-4 py-3 rounded-lg text-sm transition-all" placeholder="John Architect" type="text" required />
                </div>
                <div className="space-y-2">
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest">Email Address</label>
                  <input className="w-full bg-slate-50 border-none focus:ring-2 focus:ring-primary/20 px-4 py-3 rounded-lg text-sm transition-all" placeholder="john@example.com" type="email" required />
                </div>
              </div>
              <div className="space-y-2">
                <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest">Message</label>
                <textarea className="w-full bg-slate-50 border-none focus:ring-2 focus:ring-primary/20 px-4 py-3 rounded-lg text-sm transition-all resize-none" placeholder="Briefly describe your requirements..." rows={4} required></textarea>
              </div>
              <button 
                className="bg-primary text-white px-10 py-4 rounded-xl font-bold text-sm flex items-center justify-center gap-3 transition-transform active:scale-95 shadow-lg shadow-primary/10 disabled:opacity-70"
                disabled={loading}
              >
                {loading ? 'Sending...' : 'Send Inquiry'}
                <Send className="w-4 h-4" />
              </button>
            </form>
          )}
        </section>

        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-primary text-white p-8 rounded-2xl relative overflow-hidden">
            <h3 className="text-xl font-bold mb-6 relative z-10">Direct Contact</h3>
            <div className="space-y-6 relative z-10">
              <div className="flex items-start gap-4">
                <div className="bg-white/10 p-2 rounded-lg">
                  <Mail className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-widest opacity-60">General inquiries</p>
                  <p className="text-lg font-medium">precision@predictmyhome.ai</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="bg-white/10 p-2 rounded-lg">
                  <Phone className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-widest opacity-60">Office Line</p>
                  <p className="text-lg font-medium">+1 (555) 012-3456</p>
                </div>
              </div>
            </div>
            <div className="mt-10 flex gap-4 relative z-10">
              <Globe className="w-5 h-5 opacity-60 cursor-pointer hover:opacity-100" />
              <Share2 className="w-5 h-5 opacity-60 cursor-pointer hover:opacity-100" />
              <Network className="w-5 h-5 opacity-60 cursor-pointer hover:opacity-100" />
            </div>
          </div>

          <div className="bg-slate-100 rounded-2xl flex-grow overflow-hidden relative min-h-[300px]">
            <img src="/pic/house.jpg" alt="Map" className="w-full h-full object-cover grayscale" referrerPolicy="no-referrer" />
            <div className="absolute inset-0 bg-primary/10 mix-blend-multiply"></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
              <div className="w-12 h-12 bg-secondary rounded-full flex items-center justify-center text-white shadow-xl">
                <MapPin className="w-6 h-6" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
