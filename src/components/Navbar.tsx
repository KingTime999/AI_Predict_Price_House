import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/src/lib/utils';

export default function Navbar() {
  const location = useLocation();
  const isAuthPage = ['/login', '/register'].includes(location.pathname);

  if (isAuthPage) return null;

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Search', path: '/search' },
    { name: 'Price Prediction', path: '/predict' },
    { name: 'Services', path: '/services' },
    { name: 'About', path: '/about' },
    { name: 'Contact', path: '/contact' },
  ];

  return (
    <nav className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-md border-b border-slate-100">
      <div className="max-w-7xl mx-auto px-6 h-20 flex justify-between items-center">
        <Link to="/" className="text-2xl font-black tracking-tighter text-primary uppercase font-headline">
          PredictMyHome
        </Link>
        
        <div className="hidden md:flex items-center space-x-8">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={cn(
                "text-sm font-semibold transition-colors hover:text-primary",
                location.pathname === link.path ? "text-primary border-b-2 border-secondary pb-1" : "text-slate-500"
              )}
            >
              {link.name}
            </Link>
          ))}
        </div>

        <div className="flex items-center space-x-4">
          <Link to="/login" className="text-sm font-semibold text-slate-500 hover:text-primary transition-colors px-4 py-2">
            Login
          </Link>
          <Link to="/register" className="bg-primary text-on-primary px-6 py-2.5 rounded-xl font-semibold text-sm hover:opacity-90 transition-all active:scale-95">
            Register
          </Link>
        </div>
      </div>
    </nav>
  );
}
