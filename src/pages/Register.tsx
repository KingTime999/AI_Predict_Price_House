import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, User, Chrome } from 'lucide-react';

export default function Register() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate registration
    setTimeout(() => {
      setLoading(false);
      navigate('/dashboard');
    }, 1000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-slate-50">
      <div className="grid lg:grid-cols-12 max-w-7xl w-full bg-white rounded-2xl overflow-hidden shadow-2xl">
        <div className="lg:col-span-5 architectural-gradient p-12 flex flex-col justify-between text-white relative overflow-hidden">
          <div className="relative z-10">
            <div className="mb-12">
              <span className="text-2xl font-black tracking-tighter block mb-2">PredictMyHome</span>
              <div className="h-1 w-12 bg-secondary-container"></div>
            </div>
            <h1 className="font-headline text-5xl font-extrabold tracking-tighter leading-tight mb-8">
              The Future of <br />Property Intelligence.
            </h1>
            <div className="space-y-8">
              <div className="flex gap-4">
                <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center">
                  <User className="text-secondary-container w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-headline text-lg font-bold mb-1">Track Valuations</h3>
                  <p className="text-white/70 text-sm leading-relaxed">Monitor real-time market shifts and AI-powered price predictions.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-12 pt-8 border-t border-white/10 relative z-10">
            <p className="text-xs text-white/50 font-bold uppercase tracking-widest">Architectural Precision • Data Integrity</p>
          </div>
        </div>

        <div className="lg:col-span-7 p-12 md:p-20 bg-white">
          <div className="max-w-md mx-auto">
            <div className="mb-10">
              <h2 className="font-headline text-3xl font-extrabold text-primary mb-2">Create Account</h2>
              <p className="text-slate-500 text-sm">Join the ecosystem of digital real estate architecture.</p>
            </div>
            <form className="space-y-6" onSubmit={handleSubmit}>
              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Full Name</label>
                <div className="relative">
                  <input 
                    className="w-full bg-slate-50 px-4 py-4 rounded-lg text-sm focus:outline-none border-b-2 border-transparent focus:border-primary transition-all" 
                    placeholder="Alexander Wright" 
                    type="text" 
                    required
                  />
                  <User className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Email Address</label>
                <div className="relative">
                  <input 
                    className="w-full bg-slate-50 px-4 py-4 rounded-lg text-sm focus:outline-none border-b-2 border-transparent focus:border-primary transition-all" 
                    placeholder="architect@domain.com" 
                    type="email" 
                    required
                  />
                  <Mail className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Secure Password</label>
                <div className="relative">
                  <input 
                    className="w-full bg-slate-50 px-4 py-4 rounded-lg text-sm focus:outline-none border-b-2 border-transparent focus:border-primary transition-all" 
                    placeholder="••••••••••••" 
                    type="password" 
                    required
                  />
                  <Lock className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300 w-5 h-5" />
                </div>
              </div>
              <button 
                className="w-full py-4 bg-primary text-white font-headline font-bold rounded-xl shadow-lg hover:opacity-90 transition-all disabled:opacity-50" 
                type="submit"
                disabled={loading}
              >
                {loading ? 'Creating Account...' : 'Register'}
              </button>
            </form>
            <div className="mt-10">
              <div className="relative flex items-center justify-center mb-8">
                <div className="w-full border-t border-slate-100"></div>
                <span className="absolute px-4 bg-white text-xs font-bold text-slate-400 uppercase tracking-widest">Or Register With</span>
              </div>
              <button className="w-full flex items-center justify-center gap-2 py-3 px-4 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors">
                <Chrome className="w-5 h-5 text-primary" />
                <span className="text-sm font-semibold text-primary">Google</span>
              </button>
            </div>
            <p className="mt-12 text-center text-sm text-slate-500">
              Already have an account? <Link to="/login" className="text-secondary font-bold hover:underline ml-1">Login here</Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
