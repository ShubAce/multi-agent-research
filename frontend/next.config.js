/** @type {import('next').NextConfig} */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const nextConfig = {
	output: "standalone", // needed for Docker multi-stage build
	async rewrites() {
		return [
			{
				// Proxy /api/v1/* calls to the FastAPI backend so the frontend
				// never has to deal with CORS in development.
				source: "/api/v1/:path*",
				destination: `${API_URL}/api/v1/:path*`,
			},
		];
	},
};

module.exports = nextConfig;
