import { useState } from "react";
import { IoMenu } from "react-icons/io5";
import { IoMdAdd } from "react-icons/io";
import { IoIosHelpCircleOutline } from "react-icons/io";
import { GoHistory } from "react-icons/go";
import { IoSettingsSharp } from "react-icons/io5";
import { FaRegCircleUser } from "react-icons/fa6";
import { useNavigate } from "react-router-dom";
import useStore from "../store/store.js";

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const prevPrompt = useStore((state) => state.prevPrompt);
  const setRecentPrompt = useStore((state) => state.setRecentPrompt);
  const setUserData = useStore((state) => state.setUserData);
  const token = useStore((state) => state.accessToken);

  const loggedIn = useStore((state) => state.loggedIn);
  const setLoggedIn = useStore((state) => state.setLoggedIn);

  const navigate = useNavigate();

  const toggleSidebar = () => {
    setIsOpen((prev) => !prev);
  };

  const logOut = async () => {
    try {
      const response = await fetch("/api/users/auth/logout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });
      if (response.ok) {
        setLoggedIn(false);
        setUserData(null);

        navigate("/");
      }
    } catch (err) {
      console.log("Error Occurred: ", err.message);
    }
  };

  const register = () => {
    navigate("/register");
  };

  const openPrompt = (prompt) => {
    setRecentPrompt(prompt);
  };

  return (
    <>
      <button
        className="fixed top-1 left-1 z-50 md:hidden bg-gray-800 text-white p-2 rounded-md"
        onClick={toggleSidebar}
      >
        <IoMenu className="w-6 h-6" />
      </button>

      <div
        className={`fixed inset-y-0 left-0 bg-white w-64 shadow-md transition-transform transform ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0 md:relative md:w-64 z-40 flex flex-col h-screen`}
      >
        <div className="w-full h-20 border-b-2 flex items-center px-4">
          <button className="md:hidden" onClick={toggleSidebar}>
            <IoMenu className="w-6 h-6 text-gray-700" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4">
          <div className="w-full h-10 flex items-center gap-2 cursor-pointer">
            <p className="text-sm">New</p>
            <IoMdAdd />
          </div>

          <div className="w-full h-10 gap-2 mt-4 flex">
            <p className="text-sm">Recent</p>
          </div>
          <div className="w-full p-2">
            {prevPrompt.length > 0 ? (
              prevPrompt.map((prompt, index) => (
                <p
                  key={index}
                  className="overflow-hidden whitespace-nowrap text-sm text-gray-900 bg-slate-100 px-2 rounded-md mb-4 cursor-pointer"
                  onClick={() => openPrompt(prompt)}
                >
                  {prompt.slice(0, 18)}
                </p>
              ))
            ) : (
              <p className="text-sm text-gray-500">No recent prompts</p>
            )}
          </div>
        </div>

        <div className="w-full p-4 border-t-2">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 cursor-pointer">
              <IoIosHelpCircleOutline />
              Help
            </div>
            <div className="flex items-center gap-2 cursor-pointer">
              <IoSettingsSharp />
              Settings
            </div>
            <div className="flex items-center gap-2 cursor-pointer">
              <GoHistory />
              History
            </div>
          </div>

          <div
            className="w-full h-6 flex items-center gap-2 cursor-pointer mt-4"
            onClick={register}
          >
            <FaRegCircleUser />
            {loggedIn ? (
              <p className="text-sm" onClick={logOut}>
                Log out
              </p>
            ) : (
              <p className="text-sm" onClick={register}>
                Sign Up
              </p>
            )}
          </div>
        </div>
      </div>

      {isOpen && (
        <div
          className="fixed inset-0 bg-black opacity-50 md:hidden"
          onClick={toggleSidebar}
        ></div>
      )}
    </>
  );
};

export default Sidebar;
