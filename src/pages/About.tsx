import React from 'react';
import { motion } from 'motion/react';
import { Target, Compass, Zap, Users } from 'lucide-react';

export default function About() {
  return (
    <div className="flex flex-col">
      <section className="relative min-h-[60vh] flex items-center overflow-hidden bg-slate-50 py-24">
        <div className="max-w-7xl mx-auto px-8 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          <div className="lg:col-span-7">
            <span className="inline-block px-4 py-1.5 bg-secondary/10 text-secondary rounded-full text-xs font-bold tracking-widest uppercase mb-6">Founding Philosophy</span>
            <h1 className="text-6xl md:text-7xl font-extrabold font-headline tracking-tighter text-primary leading-[1.1] mb-8">
              The Digital <br /><span className="text-secondary">Architect.</span>
            </h1>
            <p className="text-xl text-slate-600 max-w-2xl leading-relaxed font-light">
              Where the precision of high-end architectural aesthetics meets the rigorous logic of deep data science. We don't just predict prices; we curate market intelligence.
            </p>
          </div>
          <div className="lg:col-span-5">
            <div className="aspect-[4/5] rounded-xl overflow-hidden shadow-2xl">
              <img src="https://picsum.photos/seed/about/800/1000" alt="Architecture" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
            </div>
          </div>
        </div>
      </section>

      <section className="py-32 bg-white">
        <div className="max-w-7xl mx-auto px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-24">
            <div className="space-y-8">
              <h2 className="text-sm font-bold text-slate-400 tracking-[0.2em] uppercase">The Mission</h2>
              <h3 className="text-4xl font-extrabold font-headline tracking-tight text-primary">Democratizing precision, maintaining elegance.</h3>
              <p className="text-slate-600 leading-relaxed">
                Our mission is to bridge the gap between speculative real estate and architectural truth. By leveraging neural networks and historical spatial data, we empower homeowners and investors with professional-grade clarity.
              </p>
            </div>
            <div className="space-y-8">
              <h2 className="text-sm font-bold text-slate-400 tracking-[0.2em] uppercase">The Vision</h2>
              <h3 className="text-4xl font-extrabold font-headline tracking-tight text-primary">A global standard for property valuation.</h3>
              <p className="text-slate-600 leading-relaxed">
                We envision a world where every property transaction is backed by the "Architectural Pulse"—a unified metric of value that respects both the aesthetic soul of a home and the hard data of its surroundings.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="py-32 bg-slate-50">
        <div className="max-w-7xl mx-auto px-8">
          <div className="mb-20 text-center">
            <h2 className="text-5xl font-extrabold font-headline tracking-tighter text-primary">Architectural Experts</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="group">
                <div className="aspect-[3/4] rounded-xl overflow-hidden mb-6 bg-slate-200">
                  <img src={`https://picsum.photos/seed/person-${i}/400/500`} alt="Expert" className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" referrerPolicy="no-referrer" />
                </div>
                <h4 className="text-lg font-bold font-headline text-primary">Dr. Elias Thorne</h4>
                <p className="text-sm text-slate-500 font-medium">Head of Algorithmic Logic</p>
                <div className="mt-2 h-0.5 w-8 bg-secondary group-hover:w-full transition-all duration-300"></div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
