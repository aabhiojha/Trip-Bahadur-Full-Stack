const Loading = () => {
  return (
    <div className="w-2/3 h-screen flex flex-col pr-14 pl-10 pt-4 pb-14 items-center justify-start gap-4">
      <div className="w-full h-5 rounded-md bg-gradient-to-r from-[#9ed7ff] via-white to-[#9ed7ff] bg-[800px_50px] animate-loader"></div>
      <div className="w-full h-5 rounded-md bg-gradient-to-r from-[#9ed7ff] via-white to-[#9ed7ff] bg-[800px_50px] animate-loader"></div>
      <div className="w-full h-5 rounded-md bg-gradient-to-r from-[#9ed7ff] via-white to-[#9ed7ff] bg-[800px_50px] animate-loader"></div>
    </div>
  );
};

export default Loading;
