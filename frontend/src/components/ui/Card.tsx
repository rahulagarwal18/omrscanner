// Card.tsx
const Card = ({ children }: { children: React.ReactNode }) => {
  return <div className="border p-4 rounded">{children}</div>;
};

export default Card; // ✅ default export

export const CardContent = ({ children }: { children: React.ReactNode }) => {
  return <div>{children}</div>;
};
