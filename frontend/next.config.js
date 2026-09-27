/** @type {import('next').NextConfig} */
const nextConfig = {
	// Standalone output is for the Docker image only: it creates symlinks,
	// which Windows refuses without Developer Mode (EPERM during `pnpm build`).
	output: process.env.BUILD_STANDALONE === "1" ? "standalone" : undefined,
	reactStrictMode: true,
};

module.exports = nextConfig;
