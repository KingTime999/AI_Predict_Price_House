export interface PropertyCard {
  id: string;
  title: string;
  location: string;
  price: number;
  beds: number;
  baths: number;
  sqft: number;
  image: string;
  tag?: string | null;
}

export interface PropertyDetail extends PropertyCard {
  description: string;
  images: string[];
  agent: {
    name: string;
    role: string;
    image: string;
  };
  similar: PropertyCard[];
  rawDetails: Record<string, string | number | null>;
}

export interface PropertyListResponse {
  items: PropertyCard[];
  total: number;
}

export interface PredictionPayload {
  area: number;
  location: string;
  bedrooms: number;
  bathrooms: number;
  frontage?: number;
  accessRoad?: number;
  floors?: number;
  houseDirection?: string;
  balconyDirection?: string;
  legalStatus?: string;
  furnitureState?: string;
}

export interface PredictionResult {
  estimatedValue: number;
  confidence: number;
  trend: number;
  analysis: string;
}

const API_BASE = '/api';
const DIRECT_LOCAL_API_BASE = 'http://127.0.0.1:8000/api';

function buildFallbackUrl(url: string): string | null {
  if (!url.startsWith('/api/')) return null;
  return url.replace('/api', DIRECT_LOCAL_API_BASE);
}

async function tryFetch(url: string, init?: RequestInit): Promise<Response> {
  return fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    ...init,
  });
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const fallbackUrl = buildFallbackUrl(url);

  let response: Response;
  try {
    response = await tryFetch(url, init);
  } catch (networkError) {
    if (!fallbackUrl) {
      throw networkError;
    }
    response = await tryFetch(fallbackUrl, init);
  }

  if (!response.ok && fallbackUrl && [404, 502, 503, 504].includes(response.status)) {
    response = await tryFetch(fallbackUrl, init);
  }

  if (!response.ok) {
    const message = await response.text();
    throw new Error(
      message || `Request failed with status ${response.status}. Please check backend API on port 8000.`
    );
  }

  return response.json() as Promise<T>;
}

export async function fetchProperties(params?: {
  location?: string;
  minPrice?: number;
  maxPrice?: number;
  minBeds?: number;
  limit?: number;
  skip?: number;
}): Promise<PropertyListResponse> {
  const query = new URLSearchParams();

  if (params?.location) query.set('location', params.location);
  if (params?.minPrice !== undefined) query.set('minPrice', String(params.minPrice));
  if (params?.maxPrice !== undefined) query.set('maxPrice', String(params.maxPrice));
  if (params?.minBeds !== undefined) query.set('minBeds', String(params.minBeds));
  if (params?.limit !== undefined) query.set('limit', String(params.limit));
  if (params?.skip !== undefined) query.set('skip', String(params.skip));

  const suffix = query.toString() ? `?${query.toString()}` : '';
  return request<PropertyListResponse>(`${API_BASE}/properties${suffix}`);
}

export async function fetchFeaturedProperties(limit = 3): Promise<PropertyCard[]> {
  return request<PropertyCard[]>(`${API_BASE}/properties/featured?limit=${limit}`);
}

export async function fetchPropertyById(id: string): Promise<PropertyDetail> {
  return request<PropertyDetail>(`${API_BASE}/properties/${id}`);
}

export async function predictPropertyPrice(payload: PredictionPayload): Promise<PredictionResult> {
  return request<PredictionResult>(`${API_BASE}/predict`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function submitInquiry(payload: {
  name: string;
  email: string;
  message: string;
  propertyId?: string;
}): Promise<{ message: string }> {
  return request<{ message: string }>(`${API_BASE}/inquiries`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
