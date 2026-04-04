import React from 'react';
import { Outlet } from 'react-router-dom';
import DashboardSidebar from './DashboardSidebar';
import DashboardHeader from './DashboardHeader';

export default function DashboardLayout() {
  return (
    <div className="flex min-h-screen bg-white">
      <DashboardSidebar />
      <div className="flex-1 flex flex-col">
        <DashboardHeader />
        <main className="flex-1 p-8 md:p-12 lg:p-16 overflow-y-auto">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
        <footer className="w-full border-t border-slate-100 bg-slate-50/50 py-12 px-10">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8 opacity-40">
            <p className="text-[10px] uppercase font-black tracking-widest text-primary">© 2026 PredictMyHome. Engineered for Precision.</p>
            <div className="flex gap-10">
              <button className="text-[10px] uppercase font-black tracking-widest hover:text-primary transition-colors">Privacy Policy</button>
              <button className="text-[10px] uppercase font-black tracking-widest hover:text-primary transition-colors">Terms of Service</button>
              <button className="text-[10px] uppercase font-black tracking-widest hover:text-primary transition-colors">Cookie Settings</button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
