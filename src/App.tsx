import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import SearchPage from './pages/Search';
import Predict from './pages/Predict';
import Services from './pages/Services';
import About from './pages/About';
import Contact from './pages/Contact';
import Login from './pages/Login';
import Register from './pages/Register';
import PropertyDetails from './pages/PropertyDetails';
import InquirySuccess from './pages/InquirySuccess';
import DashboardLayout from './components/DashboardLayout';
import Profile from './pages/dashboard/Profile';
import Activity from './pages/dashboard/Activity';
import Favorites from './pages/dashboard/Favorites';
import Settings from './pages/dashboard/Settings';
import Support from './pages/dashboard/Support';

function AppContent() {
  const location = useLocation();
  const isAuthPage = ['/login', '/register'].includes(location.pathname);
  const isDashboard = location.pathname.startsWith('/dashboard') || location.pathname === '/inquiry-success';

  return (
    <div className="flex flex-col min-h-screen">
      {!isAuthPage && !isDashboard && <Navbar />}
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/property/:id" element={<PropertyDetails />} />
          <Route path="/inquiry-success" element={<InquirySuccess />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/services" element={<Services />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          {/* Dashboard Routes */}
          <Route path="/dashboard" element={<DashboardLayout />}>
            <Route index element={<Navigate to="/dashboard/profile" replace />} />
            <Route path="profile" element={<Profile />} />
            <Route path="activity" element={<Activity />} />
            <Route path="favorites" element={<Favorites />} />
            <Route path="settings" element={<Settings />} />
            <Route path="support" element={<Support />} />
          </Route>
        </Routes>
      </main>
      {!isAuthPage && !isDashboard && <Footer />}
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
