export default function UploadCard({ onSelect }) {
  return (
    <label className="block p-10 bg-cardDark border border-neonPurple/40 rounded-xl shadow-neon cursor-pointer text-center hover:border-neonPurple transition">
      <p className="text-xl text-gray-300">Click to upload an image</p>
      <input
        type="file"
        className="hidden"
        accept="image/*"
        onChange={(e) => onSelect(e.target.files[0])}
      />
    </label>
  );
}
