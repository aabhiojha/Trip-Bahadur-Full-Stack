import React, { useRef } from "react";
import { FcGoogle } from "react-icons/fc";
import { ImFacebook2 } from "react-icons/im";

import { useNavigate } from "react-router-dom";

const Register = () => {
  const navigate = useNavigate();

  const fullNameElement = useRef(null);
  const usernameElement = useRef(null);
  const emailElement = useRef(null);
  const passwordElement = useRef(null);

  const submitForm = async (e) => {
    e.preventDefault();
    const fullName = fullNameElement.current.value;
    const username = usernameElement.current.value;
    const email = emailElement.current.value;
    const password = passwordElement.current.value;

    if (!fullName || !username || !email || !password) {
      alert("Please fill out all the fields");
      return;
    }

    const userData = {
      fullName,
      username,
      email,
      password,
    };

    try {
      const response = await fetch("/api/users/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        localStorage.removeItem("token");
        navigate("/login");
      } else {
        const errorData = await response.json();
        console.log("Registration failed: ", errorData.message);
        alert(errorData.message || "Something went wrong");
      }
    } catch (err) {
      console.log("Error: ", err.message);
    }
  };

  return (
    <div className="flex flex-col justify-between items-center absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white border border-gray-200 w-1/3 h-6/7 rounded-2xl px-6 shadow-lg">
      {/* Logo */}
      <div className="w-full h-1/6 flex items-center justify-center border-b border-gray-300 mb-4">
        <img src="/trekBahadur_logo.png" className="w-64 h-auto" alt="Logo" />
      </div>

      {/* Sign Up using */}
      <div className="w-full flex flex-col items-center mb-2">
        <p className="text-gray-600 mb-2">Sign Up using</p>
        <div className="flex space-x-4">
          <button className="border border-gray-400 flex items-center gap-2 px-6 py-2 rounded-lg text-lg bg-white hover:bg-gray-100 transition-all duration-200">
            <FcGoogle />
            Google
          </button>
          <button className="border border-gray-400 px-6  flex items-center gap-2 py-2 rounded-lg text-lg bg-blue-600 text-white hover:bg-blue-700 transition-all duration-200">
            <ImFacebook2 />
            Facebook
          </button>
        </div>
      </div>

      {/* Form */}
      <form className="w-full flex flex-col space-y-4 px-6">
        <div className="flex flex-col">
          <label className="text-gray-600 mb-1">Full Name</label>
          <input
            type="text"
            ref={fullNameElement}
            required
            placeholder="Full Name"
            className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-gray-600 mb-1">username</label>
          <input
            ref={usernameElement}
            required
            type="text"
            placeholder="username"
            className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-gray-600 mb-1">Email</label>
          <input
            ref={emailElement}
            type="email"
            placeholder="Email"
            required
            className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex flex-col">
          <label className="text-gray-600 mb-1">Password</label>
          <input
            ref={passwordElement}
            required
            type="password"
            placeholder="********"
            className="border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </form>

      {/* Footer */}
      <div className="w-full flex justify-between items-center">
        <div className="text-gray-600">
          Already have an account?
          <span
            onClick={() => {
              navigate("/login");
            }}
            className="text-blue-500 hover:underline cursor-pointer"
          >
            Login
          </span>
        </div>
        <button
          onClick={submitForm}
          className="bg-blue-600 text-white rounded-lg px-6 py-2 hover:bg-blue-700 transition-all duration-200"
        >
          Register
        </button>
      </div>
    </div>
  );
};

export default Register;
