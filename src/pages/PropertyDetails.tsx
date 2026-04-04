import React, { useEffect, useMemo, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import {
  Square,
  Bed,
  Bath,
  TrendingUp,
  MapPin,
  Phone,
  Calendar,
  ArrowLeft,
  ArrowRight,
  Grid,
  Mail,
  User,
  Loader2,
} from 'lucide-react';

import { fetchPropertyById, PropertyDetail as PropertyDetailType, submitInquiry } from '../services/apiService';

function formatPriceInMillionVnd(price: number): string {
  return `${price.toLocaleString('vi-VN')} tr VND`;
}

export default function PropertyDetails() {
  const { id = '' } = useParams();
  const navigate = useNavigate();

  const [property, setProperty] = useState<PropertyDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [inquiry, setInquiry] = useState({
    name: '',
    email: '',
    message: '',
  });

  useEffect(() => {
    const loadProperty = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await fetchPropertyById(id);
        setProperty(data);
      } catch (requestError) {
        setError('Cannot load property details. Please check backend API.');
        console.error(requestError);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      void loadProperty();
    }
  }, [id]);

  const gallery = useMemo(() => {
    if (!property) return [];

    const source = property.images.length ? property.images : [property.image];
    return [
      source[0] ?? property.image,
      source[1] ?? source[0] ?? property.image,
      source[2] ?? source[0] ?? property.image,
      source[3] ?? source[0] ?? property.image,
    ];
  }, [property]);

  const handleInquiryChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setInquiry((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!property) return;

    setSubmitting(true);
    try {
      await submitInquiry({
        name: inquiry.name,
        email: inquiry.email,
        message: inquiry.message,
        propertyId: property.id,
      });
      navigate('/inquiry-success');
    } catch (submitError) {
      console.error(submitError);
      alert('Failed to submit inquiry. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        <Loader2 className="mr-3 h-5 w-5 animate-spin" />
        Loading property details...
      </div>
    );
  }

  if (error || !property) {
    return (
      <div className="min-h-screen flex items-center justify-center px-8">
        <div className="max-w-xl rounded-xl border border-rose-200 bg-rose-50 p-8 text-center">
          <p className="text-rose-700 mb-4">{error || 'Property not found.'}</p>
          <Link to="/search" className="text-primary font-bold underline">
            Back to search
          </Link>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-slate-50 min-h-screen"
    >
      <section className="max-w-7xl mx-auto px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 grid-rows-2 gap-4 h-[600px] mb-12">
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="md:col-span-2 md:row-span-2 relative overflow-hidden rounded-2xl group"
          >
            <img
              className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
              src={gallery[0]}
              alt={property.title}
              referrerPolicy="no-referrer"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
            <div className="absolute bottom-8 left-8 text-white">
              <span className="bg-secondary text-white px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-[0.2em] mb-3 inline-block">
                {property.tag || 'Featured Property'}
              </span>
              <h1 className="text-5xl font-extrabold tracking-tighter font-headline">{property.title}</h1>
            </div>
          </motion.div>

          <div className="md:col-span-1 overflow-hidden rounded-2xl relative group">
            <img className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" src={gallery[1]} alt="Interior" referrerPolicy="no-referrer" />
          </div>

          <div className="md:col-span-1 overflow-hidden rounded-2xl relative group">
            <img className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" src={gallery[2]} alt="Kitchen" referrerPolicy="no-referrer" />
          </div>

          <div className="md:col-span-2 overflow-hidden rounded-2xl relative group">
            <img className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" src={gallery[3]} alt="Master Bedroom" referrerPolicy="no-referrer" />
            <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-all cursor-pointer">
              <div className="bg-white/90 px-6 py-3 rounded-full flex items-center gap-3 shadow-xl">
                <Grid className="w-5 h-5 text-primary" />
                <span className="font-bold text-primary text-xs uppercase tracking-widest">View Photos</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 items-start">
          <div className="lg:col-span-2 space-y-12">
            <div className="bg-white rounded-2xl p-8 flex flex-wrap gap-12 items-center shadow-sm border border-slate-100">
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Price</span>
                <div className="text-4xl font-black text-primary font-headline">{formatPriceInMillionVnd(property.price)}</div>
              </div>

              <div className="h-12 w-[1px] bg-slate-100 hidden sm:block"></div>

              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-slate-50 flex items-center justify-center text-primary">
                  <Square className="w-6 h-6" />
                </div>
                <div>
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Area</span>
                  <span className="text-lg font-bold">{property.sqft.toLocaleString('vi-VN')} <span className="text-xs font-medium opacity-40">m2</span></span>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-slate-50 flex items-center justify-center text-primary">
                  <Bed className="w-6 h-6" />
                </div>
                <div>
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Beds</span>
                  <span className="text-lg font-bold">{property.beds}</span>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-slate-50 flex items-center justify-center text-primary">
                  <Bath className="w-6 h-6" />
                </div>
                <div>
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Baths</span>
                  <span className="text-lg font-bold">{property.baths}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 border-l-4 border-secondary relative overflow-hidden shadow-sm">
              <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                  <h3 className="text-xl font-bold font-headline text-primary flex items-center gap-3">
                    <TrendingUp className="text-secondary w-6 h-6" />
                    Predictive Market Analysis
                  </h3>
                  <p className="text-sm text-slate-500 mt-2 max-w-md font-medium leading-relaxed">
                    This listing is integrated with our Python prediction API. Run a custom valuation in the Predict page using dataset-backed features.
                  </p>
                </div>
                <Link to="/predict" className="bg-secondary text-white px-8 py-4 rounded-xl font-bold text-xs uppercase tracking-[0.15em] hover:opacity-90 transition-all shadow-lg shadow-secondary/20 whitespace-nowrap text-center">
                  Open Predictor
                </Link>
              </div>
            </div>

            <div className="space-y-8">
              <h2 className="text-3xl font-extrabold tracking-tight font-headline text-primary">Property Narrative</h2>
              <div className="bg-slate-200 h-[1px] w-full"></div>
              <div className="text-slate-600 text-lg font-light leading-relaxed space-y-6 whitespace-pre-line">
                {property.description}
              </div>
            </div>

            <div className="space-y-6">
              <h2 className="text-xl font-bold font-headline text-primary flex items-center gap-3">
                <MapPin className="w-6 h-6" />
                Key Property Data
              </h2>
              <div className="rounded-2xl border border-slate-200 bg-white p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(property.rawDetails).map(([key, value]) => (
                    <div key={key} className="rounded-lg border border-slate-100 p-4">
                      <p className="text-[10px] uppercase tracking-widest text-slate-400 font-bold mb-1">{key}</p>
                      <p className="text-sm text-slate-700 font-semibold">{value === null ? 'N/A' : String(value)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <aside className="sticky top-8">
            <div className="bg-white p-8 rounded-2xl shadow-xl border border-slate-100">
              <div className="flex items-center gap-5 mb-10">
                <div className="w-16 h-16 rounded-full overflow-hidden border-2 border-slate-50">
                  <img className="w-full h-full object-cover" src={property.agent.image} alt={property.agent.name} referrerPolicy="no-referrer" />
                </div>
                <div>
                  <h4 className="font-bold text-primary font-headline text-lg">{property.agent.name}</h4>
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{property.agent.role}</span>
                </div>
              </div>

              <form className="space-y-6" onSubmit={handleSubmit}>
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                    <input
                      name="name"
                      value={inquiry.name}
                      onChange={handleInquiryChange}
                      className="w-full bg-slate-50 border-none rounded-xl focus:ring-2 focus:ring-primary/5 pl-12 py-4 font-medium transition-all text-sm"
                      placeholder="Enter your name"
                      type="text"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Email Address</label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                    <input
                      name="email"
                      value={inquiry.email}
                      onChange={handleInquiryChange}
                      className="w-full bg-slate-50 border-none rounded-xl focus:ring-2 focus:ring-primary/5 pl-12 py-4 font-medium transition-all text-sm"
                      placeholder="email@example.com"
                      type="email"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Inquiry Message</label>
                  <textarea
                    name="message"
                    value={inquiry.message}
                    onChange={handleInquiryChange}
                    className="w-full bg-slate-50 border-none rounded-xl focus:ring-2 focus:ring-primary/5 px-4 py-4 font-medium transition-all text-sm resize-none"
                    placeholder="I am interested in viewing this property..."
                    rows={4}
                    required
                  ></textarea>
                </div>

                <button
                  className="w-full bg-primary text-white py-5 rounded-xl font-bold tracking-[0.1em] text-xs uppercase hover:bg-primary/90 transition-all active:scale-[0.98] shadow-xl shadow-primary/20 disabled:opacity-70"
                  disabled={submitting}
                >
                  {submitting ? 'Submitting...' : 'Submit Inquiry'}
                </button>
              </form>

              <div className="mt-10 pt-10 border-t border-slate-100 space-y-4">
                <button className="flex items-center gap-4 text-xs font-bold text-primary hover:text-secondary transition-colors w-full uppercase tracking-widest">
                  <Phone className="w-4 h-4" />
                  Request Direct Callback
                </button>
                <button className="flex items-center gap-4 text-xs font-bold text-primary hover:text-secondary transition-colors w-full uppercase tracking-widest">
                  <Calendar className="w-4 h-4" />
                  Schedule Virtual Tour
                </button>
              </div>
            </div>
          </aside>
        </div>

        <section className="mt-32">
          <div className="flex items-center justify-between mb-12">
            <h2 className="text-3xl font-extrabold tracking-tight font-headline text-primary">Contextual Discoveries</h2>
            <div className="flex gap-4">
              <button className="w-12 h-12 rounded-full border border-slate-200 flex items-center justify-center hover:bg-primary hover:text-white transition-all text-primary">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <button className="w-12 h-12 rounded-full border border-slate-200 flex items-center justify-center hover:bg-primary hover:text-white transition-all text-primary">
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {property.similar.map((item) => (
              <Link key={item.id} to={`/property/${item.id}`} className="bg-white rounded-2xl overflow-hidden group shadow-sm hover:shadow-xl transition-all duration-500 border border-slate-100">
                <div className="h-72 overflow-hidden relative">
                  <img className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110" src={item.image} alt={item.title} referrerPolicy="no-referrer" />
                  {item.tag && (
                    <div className="absolute top-6 right-6 bg-white/90 backdrop-blur-md px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest text-primary shadow-lg">
                      {item.tag}
                    </div>
                  )}
                </div>
                <div className="p-8">
                  <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 block">{item.location}</span>
                  <h3 className="text-2xl font-extrabold text-primary font-headline mb-6 tracking-tight">{item.title}</h3>
                  <div className="flex justify-between items-center pt-6 border-t border-slate-50">
                    <span className="text-xl font-black text-primary">{formatPriceInMillionVnd(item.price)}</span>
                    <div className="flex items-center gap-3 text-slate-400">
                      <Bed className="w-4 h-4" />
                      <span className="text-xs font-bold">{item.beds}</span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      </section>
    </motion.div>
  );
}
