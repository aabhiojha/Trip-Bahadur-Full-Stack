import { Outlet } from "react-router-dom";
import useStore from "../store/store.js";
import { useEffect } from "react";
import { motion } from "framer-motion";
import "./App.css";
const App = () => {
  const initializeAuth = useStore((state) => state.initializeAuth);
  useEffect(() => {
    initializeAuth();
  }, [initializeAuth]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 1.7 }}
    >
      <Outlet />
    </motion.div>
  );
};

export default App;
