import React from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { 
  User, 
  History, 
  Heart, 
  Settings, 
  HelpCircle, 
  BarChart3, 
  LogOut,
  Sparkles
} from 'lucide-react';

const NAV_ITEMS = [
  { icon: User, label: 'Personal Info', path: '/dashboard/profile' },
  { icon: History, label: 'Activity History', path: '/dashboard/activity' },
  { icon: Heart, label: 'Favorites', path: '/dashboard/favorites' },
  { icon: Settings, label: 'Account Settings', path: '/dashboard/settings' },
  { icon: HelpCircle, label: 'Support', path: '/dashboard/support' },
];

export default function DashboardSidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <aside className="hidden lg:flex flex-col w-72 border-r border-slate-100 bg-slate-50/50 p-8 h-screen sticky top-0">
      <div className="px-2 mb-12">
        <div className="w-14 h-14 rounded-2xl bg-primary-container flex items-center justify-center mb-6 overflow-hidden border-2 border-white shadow-sm">
          <img 
            alt="Alex Architect" 
            className="w-full h-full object-cover" 
            src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&q=80&w=200" 
            referrerPolicy="no-referrer"
          />
        </div>
        <h2 className="font-headline font-bold text-xl text-primary tracking-tight">Alex Architect</h2>
        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mt-1">Premium Member</p>
      </div>

      <nav className="flex-1 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <button 
              key={item.label}
              onClick={() => navigate(item.path)}
              className={`flex items-center gap-4 w-full px-4 py-3.5 rounded-xl text-sm font-bold transition-all group ${
                isActive 
                  ? 'bg-white text-primary shadow-sm ring-1 ring-slate-100 border-l-4 border-secondary' 
                  : 'text-slate-400 hover:text-primary hover:bg-white/50'
              }`}
            >
              <item.icon className={`w-5 h-5 ${isActive ? 'text-secondary' : 'group-hover:text-primary'}`} />
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="mt-auto pt-8 border-t border-slate-100 space-y-6">
        <button 
          onClick={() => navigate('/predict')}
          className="w-full bg-secondary text-white py-4 rounded-xl text-xs font-black uppercase tracking-widest hover:opacity-90 transition-all shadow-xl shadow-secondary/20 flex items-center justify-center gap-2"
        >
          <BarChart3 className="w-4 h-4" />
          Get New Prediction
        </button>
        <button 
          onClick={() => navigate('/login')}
          className="flex items-center gap-4 px-4 w-full text-left text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors"
        >
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </aside>
  );
}
