import { Routes, Route } from 'react-router'
import Layout from './components/Layout'
import Home from './pages/Home'
import Architecture from './pages/Architecture'
import Experiments from './pages/Experiments'
import Tasks from './pages/Tasks'
import Interpretability from './pages/Interpretability'
import Reproducibility from './pages/Reproducibility'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/interpretability" element={<Interpretability />} />
        <Route path="/reproducibility" element={<Reproducibility />} />
      </Routes>
    </Layout>
  )
}
