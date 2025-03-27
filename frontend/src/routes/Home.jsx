import Main from "../components/Main";
import Sidebar from "../components/Sidebar";

const Home = () => {
  return (
    <div className="flex flex-row justify-start">
      <Sidebar />
      <Main />
    </div>
  );
};

export default Home;
