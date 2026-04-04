import React from 'react';
import { motion } from 'motion/react';
import { Zap, ShieldCheck, Building2, UserCheck, Camera, Video, MousePointerClick, Quote, ArrowRight } from 'lucide-react';

export default function Services() {
  return (
    <div className="flex flex-col">
      <section className="py-24 px-8 max-w-7xl mx-auto">
        <span className="bg-secondary-container text-secondary px-3 py-1 rounded-full text-xs font-bold tracking-widest uppercase mb-6 inline-block">Core Capabilities</span>
        <h1 className="text-6xl md:text-8xl font-extrabold tracking-tighter font-headline text-primary max-w-4xl leading-[0.9] mb-8">Architectural Precision in Real Estate.</h1>
        <p className="text-slate-600 text-xl max-w-2xl font-light leading-relaxed">We merge deep architectural insights with AI-driven market intelligence to redefine property value and brokerage excellence.</p>
      </section>

      <section className="max-w-7xl mx-auto px-8 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          <div className="md:col-span-8 bg-white p-10 rounded-2xl prediction-pulse flex flex-col justify-between min-h-[400px] shadow-sm">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <Zap className="text-secondary w-8 h-8 fill-secondary" />
                <span className="font-headline text-sm font-bold tracking-widest uppercase text-secondary">Advanced Technology</span>
              </div>
              <h2 className="text-4xl font-extrabold font-headline text-primary mb-4 tracking-tight">AI-Powered Price Prediction</h2>
              <p className="text-slate-600 text-lg max-w-xl leading-relaxed">Our proprietary neural networks analyze over 200 data points—from local zoning changes to seasonal market shifts—delivering accuracy within 1.5% of final sale price.</p>
            </div>
            <div className="mt-8 flex gap-4">
              <div className="bg-slate-50 px-6 py-3 rounded-xl border border-slate-100">
                <span className="text-xs font-bold text-slate-400 block uppercase tracking-tighter">Accuracy Rate</span>
                <span className="text-2xl font-black text-secondary font-headline">98.5%</span>
              </div>
            </div>
          </div>

          <div className="md:col-span-4 bg-primary text-white p-10 rounded-2xl flex flex-col justify-between group overflow-hidden relative">
            <div className="relative z-10">
              <UserCheck className="text-secondary-container w-10 h-10 mb-6" />
              <h2 className="text-3xl font-extrabold font-headline mb-4 tracking-tight">Real Estate Brokerage</h2>
              <p className="text-white/70 text-sm leading-relaxed mb-6">Exclusive listings and buyer representation backed by data, not just intuition.</p>
              <ul className="space-y-3">
                <li className="flex items-center gap-2 text-sm font-semibold">
                  <ShieldCheck className="text-secondary-container w-4 h-4" />
                  Off-Market Access
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
