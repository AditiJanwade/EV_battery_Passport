import React, { useState, useEffect } from 'react';

import axios from 'axios';
import {
    LayoutDashboard, Battery, ShieldCheck, User, LogOut,
    Edit, Info, Download, MessageSquare, Menu, X,
    Zap, Thermometer, Activity, TrendingUp, DollarSign, Layers,
    AlertTriangle, Calendar, FileText, CheckCircle, Moon, Sun
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import ChatBot from './components/ChatBot';
import { generatePDF } from './utils/pdfGenerator';

function App() {
    useEffect(() => {

        const savedUser = localStorage.getItem("userId");

        if (savedUser) {
            setUserId(savedUser);
            checkUserVehicles(savedUser);
        }

    }, []);


    const [userId, setUserId] = useState("");
    const [vehicles, setVehicles] = useState([]);
    const [selectedVehicle, setSelectedVehicle] = useState(null);
    const [userVerified, setUserVerified] = useState(false);
    const [showRegisterForm, setShowRegisterForm] = useState(false);
    const [vehicleId, setVehicleId] = useState("");
    const [batteryTypeReg, setBatteryTypeReg] = useState("");
    const [buyingPrice, setBuyingPrice] = useState("");
    const [buyingDate, setBuyingDate] = useState("");
    const [manufactureDate, setManufactureDate] = useState("");
    const [showDropdown, setShowDropdown] = useState(false);
    const [formData, setFormData] = useState({
        battery_type: 'Li-ion (NMC)',
        total_dist_km: '12500',
        charging_time_min: '45'
    });

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isSidebarOpen, setSidebarOpen] = useState(true);
    const [currentView, setCurrentView] = useState('dashboard');
    const [darkMode, setDarkMode] = useState(false);

    const [profileData, setProfileData] = useState({
        vehicleId: 'EV-8842-AX',
        manufactureDate: 'Feb 2021',
        lastService: '15/10/2025',
        softwareVersion: 'BMS v4.2.1',
        serviceNote: 'Routine Checkup'
    });

    const [specs, setSpecs] = useState({
        manufacturer: "",
        architecture: "",
        capacity: "",
        cooling: "",
        cycles: "",
        warranty: ""
    });


    const handleProfileChange = (e) => {
        setProfileData({ ...profileData, [e.target.name]: e.target.value });
    };

    const toggleTheme = () => setDarkMode(!darkMode); // 'dashboard', 'batteries', 'warranty', 'profile'

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const registerVehicle = async () => {

        try {

            const res = await axios.post(
                "http://localhost:8000/register_vehicle",
                {
                    user_id: userId,
                    vehicle_id: vehicleId,
                    battery_type: batteryTypeReg,
                    buying_price: parseFloat(buyingPrice),
                    buying_date: buyingDate,
                    manufacture_date: manufactureDate
                }
            );

            alert(res.data.message);

            // ⭐ VERY IMPORTANT ADD THIS LINE
            localStorage.setItem("userId", userId);
            loadVehicles();
            // ⭐ CLEAR FORM
            setVehicleId("");
            setBatteryTypeReg("");
            setBuyingPrice("");
            setBuyingDate("");
            setManufactureDate("");
            setShowRegisterForm(false);

        } catch (err) {
            alert("Vehicle Registration Failed");
        }
    };
    const checkUserVehicles = async (uid) => {

        const id = uid || userId;

        if (!id) {
            alert("Enter User ID First");
            return;
        }

        try {

            const res = await axios.get(
                `http://localhost:8000/get_vehicles/${id}`
            );

            if (res.data.vehicles.length > 0) {

                setVehicles(res.data.vehicles);
                setUserVerified(true);
                setShowRegisterForm(false);

            } else {

                setUserVerified(true);
                setShowRegisterForm(true);

            }

            localStorage.setItem("userId", id);

        } catch (err) {

            alert("User Check Failed");

        }
    };
    const loadVehicles = async () => {

        const uid = localStorage.getItem("userId");

        if (!uid) return;

        try {
            const res = await axios.get(
                `http://localhost:8000/get_vehicles/${uid}`
            );

            setVehicles(res.data.vehicles);
        }
        catch (err) {
            console.log("Vehicle Load Error");
        }
    };


    const handleSubmit = async () => {

        if (!selectedVehicle) {
            alert("Please Select Vehicle First!");
            return;
        }

        try {

            const response = await axios.post(
                "http://localhost:8000/predict",
                {
                    user_id: userId,
                    vehicle_id: selectedVehicle.vehicle_id,
                    battery_type: selectedVehicle.battery_type,
                    total_dist_km: parseFloat(formData.total_dist_km),
                    charging_time_min: parseFloat(formData.charging_time_min)
                });

            setResult(response.data);

        } catch (err) {
            alert("Prediction Failed");
        }
    };

    const updateVehicle = async () => {

        try {

            const res = await axios.post(
                "http://localhost:8000/update_vehicle",
                {
                    user_id: userId,
                    vehicle_id: selectedVehicle.vehicle_id,
                    battery_type: batteryTypeReg,
                    buying_price: parseFloat(buyingPrice),
                    buying_date: buyingDate,
                    manufacture_date: manufactureDate
                }
            );

            alert(res.data.message);
            loadVehicles();

        } catch {
            alert("Update Failed");
        }
    };


    // Semi-circle Gauge Component
    const Gauge = ({ value, label, color }) => {
        const data = [
            { name: 'Value', value: value },
            { name: 'Empty', value: 100 - value }
        ];
        return (
            <div className="flex flex-col items-center justify-center p-4 bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 h-full relative transition-colors duration-200">
                <div className="absolute top-4 right-4 text-slate-300 dark:text-slate-500 cursor-help"><Info size={16} /></div>
                <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">{label}</h3>
                <div className="h-32 w-48 relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="70%"
                                startAngle={180}
                                endAngle={0}
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={0}
                                dataKey="value"
                                stroke="none"
                            >
                                <Cell fill={color} className="drop-shadow-md" />
                                <Cell fill="#334155" />
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-center pb-2">
                        <span className="text-3xl font-bold text-slate-800 dark:text-white">{value.toFixed(0)}%</span>
                        <p className="text-xs text-slate-400">{value > 80 ? 'Good' : 'Average'}</p>
                    </div>
                </div>
            </div>
        );
    };

    const batterySpecs = {
        'Li-ion (NMC)': {
            manufacturer: 'Panasonic / LG Chem',
            architecture: '400V System, 96s2p',
            capacity: '64.0 kWh',
            cooling: 'Active Liquid Cooling (Glycol)',
            cycles: '~1500 Cycles @ 80% SOH',
            warranty: '8 Years / 160,000 km'
        },
        'LiFePO4': { // Changed from 'LiFePO4 (LFP)' to match select option value
            manufacturer: 'CATL / BYD',
            architecture: '400V System, 106s2p',
            capacity: '60.0 kWh', // LFP has slightly lower energy density
            cooling: 'Liquid Cooling (Plate)',
            cycles: '~3000 Cycles @ 80% SOH', // LFP lasts longer
            warranty: '10 Years / 200,000 km'
        },
        'NCM_Type1': {
            manufacturer: 'Experimental / R&D',
            architecture: '800V High Voltage',
            capacity: '75.0 kWh',
            cooling: 'Immersion Cooling',
            cycles: '~1200 Cycles @ 80% SOH',
            warranty: 'Research Prototype'
        }
    };

    const currentSpecs = batterySpecs[formData.battery_type] || batterySpecs['Li-ion (NMC)'];

    // Dynamic Warranty Logic
    const getWarrantyPlans = (data) => {
        if (!data) return [
            { name: "Basic Care", price: "$499/yr", features: ['BMS Software Updates', 'Annual Health Check', '24/7 Roadside Assist'], recommended: false, disabled: true },
            { name: "Platinum Shield", price: "$999/yr", features: ['Full Battery Replacement', 'Degradation Coverage', 'Free Towing'], recommended: true, disabled: true },
            { name: "Resale Boost", price: "$199", features: ['Health Certificate', 'Transferable Warranty'], recommended: false, disabled: true }
        ];

        const soh = data.predicted_soh;
        const isHighRisk = data.risk_rating === 'High Risk';
        const isAnomaly = data.anomaly_warning;

        // Base Logic
        let plans = [
            {
                name: "Basic Care",
                price: "$499/yr",
                features: ['BMS Software Updates', 'Annual Health Check', '24/7 Roadside Assist'],
                recommended: false,
                disabled: false
            },
            {
                name: "Platinum Shield",
                price: "$999/yr", // Default
                features: ['Full Battery Replacement', 'Degradation > 30% Coverage', 'Free Towing', 'Loaner Vehicle'],
                recommended: true,
                disabled: false,
                note: ""
            },
            {
                name: "Resale Boost",
                price: "$199",
                features: ['Certified Health Certificate', 'Transferable Warranty', 'Listing Highlight'],
                recommended: false,
                disabled: false
            }
        ];

        // Dynamic Adjustments
        if (isAnomaly || isHighRisk) {
            // Severe issues: Disqualify from Insurance
            plans[1].disabled = true;
            plans[1].note = "Not eligible due to detected risk.";
            plans[1].recommended = false;

            plans[0].recommended = true; // Recommend Basic Care instead
            plans[0].note = "Recommended for monitoring.";
        } else if (soh > 90) {
            // Excellent Health: Discount
            plans[1].price = "$899/yr";
            plans[1].note = "Healthy Battery Discount Applied!";
        } else if (soh < 80) {
            // Aging Health: Premium Increase
            plans[1].price = "$1,299/yr";
            plans[1].note = "High Mileage Premium";
        }

        return plans;
    };
    const profileInput =
        "p-3 bg-slate-700/50 border border-slate-600 text-white placeholder-slate-400 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 transition w-full";

    const renderContent = () => {
        switch (currentView) {
            case 'batteries':
                return (
                    <div className="space-y-6">
                        <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 transition-colors duration-200">
                            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center">
                                <Battery className="mr-3 text-teal-500" /> Battery Specification
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <DetailRow label="Cell Manufacturer" value={currentSpecs.manufacturer} />
                                    <DetailRow label="Pack Architecture" value={currentSpecs.architecture} />
                                    <h4 className="text-slate-400 text-sm mt-4">Vehicle Purchase Date</h4>
                                    <input
                                        type="date"
                                        value={buyingDate}
                                        onChange={(e) => setBuyingDate(e.target.value)}
                                        className="bg-slate-700 text-white p-3 rounded-xl w-full mt-1"
                                    />

                                    <h4 className="text-slate-400 text-sm mt-4">Battery Manufacturing Date</h4>
                                    <input
                                        type="date"
                                        value={manufactureDate}
                                        onChange={(e) => setManufactureDate(e.target.value)}
                                        className="bg-slate-700 text-white p-3 rounded-xl w-full mt-1"
                                    />

                                    <h4 className="text-slate-400 text-sm mt-4">Vehicle Buying Price (₹)</h4>
                                    <input
                                        type="number"
                                        value={buyingPrice}
                                        onChange={(e) => setBuyingPrice(e.target.value)}
                                        className="bg-slate-700 text-white p-3 rounded-xl w-full mt-1"
                                    />
                                    <DetailRow label="Rated Capacity" value={currentSpecs.capacity} />
                                    <DetailRow label="Chemistry" value={formData.battery_type} />
                                </div>
                                <div className="space-y-4">
                                    <DetailRow label="Cooling System" value={currentSpecs.cooling} />
                                    <DetailRow label="Max Charging Power" value="150 kW (DC Fast)" />
                                    <DetailRow label="Cycle Life Rating" value={currentSpecs.cycles} />
                                    <DetailRow label="Warranty Period" value={currentSpecs.warranty} />
                                    <DetailRow label="Risk Assessment" value={result ? result.risk_rating : 'Run Analysis'} highlight={result && result.risk_rating === 'High Risk'} />
                                </div>
                            </div>
                        </div>
                        <div className="flex justify-end mt-6">
                            <button
                                onClick={updateVehicle}
                                className="bg-teal-500 px-6 py-2 rounded-xl hover:bg-teal-600"
                            >
                                Save Changes
                            </button>
                        </div>
                        {/* Additional Battery Info */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <InfoCard
                                title="Last Service"
                                value={profileData.lastService}
                                sub={profileData.serviceNote}
                                icon={<Calendar className="text-blue-500" />}
                            />
                            <InfoCard
                                title="Software Version"
                                value={profileData.softwareVersion}
                                sub="Up to date"
                                icon={<FileText className="text-purple-500" />}
                            />
                            <InfoCard
                                title="Total Energy Throughput"
                                value={result ? `${(result.latent_features.pred_charging_cycles * parseFloat(currentSpecs.capacity)).toFixed(0)} kWh` : '--'}
                                sub="Lifetime discharged"
                                icon={<Zap className="text-yellow-500" />}
                            />
                        </div>
                    </div>

                );

            case 'profile':
                return (
                    <div className="space-y-6">

                        {/* STEP 1 : USER LOGIN */}
                        {!userVerified && (
                            <div className="bg-slate-800 p-6 rounded-2xl">

                                <h3 className="text-white mb-3 text-lg">
                                    Enter User ID
                                </h3>

                                <input
                                    value={userId}
                                    onChange={(e) => setUserId(e.target.value)}
                                    className={profileInput}
                                />

                                <button
                                    onClick={() => checkUserVehicles()}
                                    className="mt-4 bg-teal-500 px-6 py-2 rounded-xl"
                                >
                                    Continue
                                </button>

                            </div>
                        )}

                        {/* STEP 2 : VEHICLE LIST */}
                        {userVerified && vehicles.length > 0 && (

                            <div className="bg-slate-800 p-6 rounded-2xl">

                                <h3 className="text-white mb-3">
                                    Registered Vehicles
                                </h3>

                                {vehicles.map(v => (
                                    <div
                                        key={v.vehicle_id}
                                        onClick={() => {
                                            setSelectedVehicle(v)
                                            setBuyingPrice(v.buying_price)
                                            setBuyingDate(v.buying_date)
                                            setManufactureDate(v.manufacture_date)
                                            setBatteryTypeReg(v.battery_type)
                                        }}
                                        className={`p-3 rounded-xl mt-2 cursor-pointer
                        ${selectedVehicle?.vehicle_id === v.vehicle_id
                                                ? "bg-teal-600"
                                                : "bg-slate-700 hover:bg-slate-600"}
                        `}
                                    >
                                        <p className="text-white">
                                            {v.vehicle_id}
                                        </p>
                                        <p className="text-xs text-slate-400">
                                            {v.battery_type}
                                        </p>
                                    </div>
                                ))}

                                <button
                                    onClick={() => setShowRegisterForm(true)}
                                    className="mt-4 bg-teal-500 px-4 py-2 rounded-xl"
                                >
                                    Add New Vehicle
                                </button>

                            </div>
                        )}

                        {/* STEP 3 : REGISTER FORM */}
                        {(showRegisterForm || vehicles.length === 0) && userVerified && (

                            <div className="bg-slate-800 p-6 rounded-2xl">

                                <h3 className="text-white mb-3">
                                    Register Vehicle
                                </h3>

                                <input
                                    placeholder="Vehicle ID"
                                    className={profileInput}
                                    onChange={(e) => setVehicleId(e.target.value)}
                                />

                                <select
                                    className={profileInput + " mt-2"}
                                    onChange={(e) => setBatteryTypeReg(e.target.value)}
                                >
                                    <option>Select Battery</option>
                                    <option value="Li-ion">Li-ion</option>
                                    <option value="LiFePO4">LFP</option>
                                    <option value="NCM_Type1">NCM Type 1</option>
                                </select>

                                <input
                                    type="number"
                                    placeholder="Buying Price"
                                    className={profileInput + " mt-2"}
                                    onChange={(e) => setBuyingPrice(e.target.value)}
                                />

                                <input
                                    type="date"
                                    className={profileInput + " mt-2"}
                                    onChange={(e) => setBuyingDate(e.target.value)}
                                />

                                <input
                                    type="date"
                                    className={profileInput + " mt-2"}
                                    onChange={(e) => setManufactureDate(e.target.value)}
                                />

                                <button
                                    onClick={registerVehicle}
                                    className="mt-4 bg-teal-500 px-6 py-2 rounded-xl"
                                >
                                    Register Vehicle
                                </button>

                            </div>
                        )}

                    </div>
                );
            case 'warranty':
                return (
                    <div className="space-y-6">
                        <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 transition-colors duration-200">
                            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center">
                                <ShieldCheck className="mr-3 text-teal-500" /> Warranty Extension Analysis
                            </h2>

                            <div className="flex items-center space-x-6 mb-8 p-6 bg-slate-50 dark:bg-slate-700/50 rounded-2xl">
                                <div className={`p-4 rounded-full ${result?.anomaly_warning ? 'bg-red-100 dark:bg-red-900/30' : 'bg-green-100 dark:bg-green-900/30'}`}>
                                    {result?.anomaly_warning ? <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" /> : <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400" />}
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold text-slate-800 dark:text-white">
                                        {result ? (result.anomaly_warning ? 'Not Eligible for Auto-Approval' : 'Pre-Approved for Platinum Warranty') : 'Analysis Required'}
                                    </h3>
                                    <p className="text-slate-500 dark:text-slate-400">
                                        {result ? (result.anomaly_warning ? 'Detected degradation patterns exceed the safe threshold for automated warranty extension.' : 'Your battery health is excellent. You qualify for our minimal deductible plan.') : 'Please run the analysis on the dashboard to check eligibility.'}
                                    </p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                {getWarrantyPlans(result).map((plan, index) => (
                                    <PlanCard
                                        key={index}
                                        name={plan.name}
                                        price={plan.price}
                                        features={plan.features}
                                        recommended={plan.recommended}
                                        disabled={plan.disabled}
                                        note={plan.note}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>
                );
            case 'dashboard':
            default:
                return (
                    <>
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">

                            {/* General Details (Input Form) */}
                            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 transition-colors duration-200">
                                <div className="flex justify-between items-center mb-6">
                                    <h3 className="font-semibold text-slate-700 dark:text-slate-400 uppercase text-xs tracking-wider">General Details</h3>
                                    <button onClick={handleSubmit} className="text-teal-500 text-sm font-medium hover:text-teal-600 flex items-center">
                                        <Edit size={14} className="mr-1" /> Run Analysis
                                    </button>
                                </div>

                                <div className="space-y-4 text-sm">
                                    <div className="flex justify-between items-center py-2 border-b border-slate-50 dark:border-slate-700">
                                        <span className="text-slate-500 dark:text-slate-400">Chemistry</span>
                                        <div className="flex justify-between items-center py-2 border-b border-slate-50 dark:border-slate-700">
                                            <span className="text-slate-500 dark:text-slate-400">

                                            </span>

                                            <span className="text-right font-medium text-slate-700 dark:text-slate-200">
                                                {selectedVehicle?.battery_type || "--"}
                                            </span>
                                        </div>

                                    </div>
                                    <div className="flex justify-between items-center py-2 border-b border-slate-50 dark:border-slate-700">
                                        <span className="text-slate-500 dark:text-slate-400">Odometer (km)</span>
                                        <input name="total_dist_km" type="number" value={formData.total_dist_km} onChange={handleChange} className="text-right font-medium text-slate-700 dark:text-slate-200 w-24 bg-slate-50 dark:bg-slate-700 rounded px-2 focus:outline-none focus:ring-1 focus:ring-teal-500" />
                                    </div>
                                    <div className="flex justify-between items-center py-2 border-b border-slate-50 dark:border-slate-700">
                                        <span className="text-slate-500 dark:text-slate-400">Charge Duration (min)</span>
                                        <input name="charging_time_min" type="number" value={formData.charging_time_min} onChange={handleChange} className="text-right font-medium text-slate-700 dark:text-slate-200 w-24 bg-slate-50 dark:bg-slate-700 rounded px-2 focus:outline-none focus:ring-1 focus:ring-teal-500" />
                                    </div>
                                    <div className="flex justify-between items-center py-2">
                                        <span className="text-slate-500 dark:text-slate-400">Risk Rating</span>
                                        <span className={`px-2 py-1 rounded text-xs font-semibold ${result?.risk_rating === 'High Risk' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'}`}>
                                            {result ? result.risk_rating : 'Unknown'}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Resale Value */}
                            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col justify-between group relative transition-colors duration-200">
                                <div className="absolute top-4 right-4 text-slate-300 dark:text-slate-600 cursor-help" title="Based on linear depreciation (10%/yr) and Battery SOH penalty."><Info size={16} /></div>
                                <div>
                                    <h3 className="font-semibold text-slate-700 dark:text-slate-400 uppercase text-xs tracking-wider mb-2">Resale Value</h3>
                                    <p className="text-slate-400 dark:text-slate-500 text-xs mb-4">Estimated market value adjusting for SOH.</p>
                                </div>
                                <div>
                                    <div className="text-4xl font-bold text-slate-800 dark:text-white flex items-start">
                                        <span className="text-xl mt-1 text-slate-400">$</span>
                                        {result ? result.resale_value_usd.toLocaleString() : '---'}
                                    </div>
                                    <div className="text-sm text-green-500 mt-2 flex items-center">
                                        <TrendingUp size={16} className="mr-1" />
                                        Market Estimate
                                    </div>
                                </div>
                            </div>

                            {/* Material Value */}
                            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 relative transition-colors duration-200">
                                <div className="absolute top-4 right-4 text-slate-300 dark:text-slate-600 cursor-help" title="Based on Argonne National Lab (BatPaC) composition models for 60kWh pack."><Info size={16} /></div>
                                <h3 className="font-semibold text-slate-700 dark:text-slate-400 uppercase text-xs tracking-wider mb-2">Material Composition</h3>
                                <p className="text-slate-400 dark:text-slate-500 text-xs mb-6">Recoverable mass (Chemistry: {formData.battery_type})</p>

                                <div className="grid grid-cols-3 gap-4 text-center">
                                    <MaterialStat label="Lithium" value={result ? result.material_composition.lithium_g : '--'} unit="g" color="text-teal-600 dark:text-teal-400" />
                                    {/* Handle LFP/NMC differences dynamically */}
                                    {result && result.material_composition.iron_g ? (
                                        <MaterialStat label="Iron" value={result.material_composition.iron_g} unit="g" color="text-gray-600 dark:text-gray-400" />
                                    ) : (
                                        <MaterialStat label="Nickel" value={result ? result.material_composition.nickel_g : '--'} unit="g" color="text-blue-600 dark:text-blue-400" />
                                    )}
                                    <MaterialStat label="Cobalt" value={result ? result.material_composition.cobalt_g : '--'} unit="g" color="text-purple-600 dark:text-purple-400" />
                                </div>
                            </div>
                        </div>

                        {/* Middle Row: Gauges */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                            <Gauge label="State of Charge (Est)" value={result ? result.estimated_soc : 0} color="#f59e0b" />
                            <Gauge label="State of Health (SOH)" value={result ? result.predicted_soh : 0} color={result?.predicted_soh > 80 ? "#22c55e" : "#ef4444"} />

                            {/* Health Status / Virtual Sensors */}
                            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 transition-colors duration-200">
                                <div className="flex justify-between items-start mb-4">
                                    <h3 className="font-semibold text-slate-700 dark:text-slate-400 uppercase text-xs tracking-wider">Internal Diagnostics</h3>
                                    <Info size={16} className="text-slate-300 dark:text-slate-600" />
                                </div>

                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-3">
                                            <div className="p-2 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg"><Activity size={18} className="text-indigo-500 dark:text-indigo-400" /></div>
                                            <div>
                                                <p className="text-sm font-medium text-slate-700 dark:text-slate-300">Est. Cycles</p>
                                                <p className="text-xs text-slate-400 dark:text-slate-500">Total charge cycles</p>
                                            </div>
                                        </div>
                                        <span className="text-lg font-bold text-slate-800 dark:text-white">{result ? result.latent_features.pred_charging_cycles.toFixed(0) : '--'}</span>
                                    </div>

                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-3">
                                            <div className="p-2 bg-orange-50 dark:bg-orange-900/30 rounded-lg"><Thermometer size={18} className="text-orange-500 dark:text-orange-400" /></div>
                                            <div>
                                                <p className="text-sm font-medium text-slate-700 dark:text-slate-300">Thermal Stress</p>
                                                <p className="text-xs text-slate-400 dark:text-slate-500">Internal temperature</p>
                                            </div>
                                        </div>
                                        <span className="text-lg font-bold text-slate-800 dark:text-white">{result ? result.latent_features.pred_battery_temp.toFixed(1) : '--'}°C</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Bottom Area: Warranty or Logs */}
                        <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 transition-colors duration-200">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="font-semibold text-slate-700 dark:text-slate-400 uppercase text-xs tracking-wider">Extended Warranty Eligibility</h3>
                                <button onClick={() => setCurrentView('warranty')} className="text-teal-500 text-sm font-medium hover:underline">View Analysis</button>
                            </div>

                            <div className="flex items-center space-x-4">
                                <div className={`p-4 rounded-full ${result?.anomaly_warning ? 'bg-red-50 dark:bg-red-900/30' : 'bg-green-50 dark:bg-green-900/30'}`}>
                                    {result?.anomaly_warning ? <AlertTriangle className="text-red-500 dark:text-red-400" /> : <ShieldCheck className="text-green-500 dark:text-green-400" />}
                                </div>
                                <div>
                                    <h4 className="font-bold text-slate-800 dark:text-white text-lg">
                                        {result ? (result.anomaly_warning ? 'Not Eligible for Auto-Extension' : 'Eligible for Platinum Coverage') : 'Analysis Required'}
                                    </h4>
                                    <p className="text-slate-500 dark:text-slate-400 text-sm max-w-2xl">
                                        {result ? (result.anomaly_warning
                                            ? 'Anomaly detected in degradation patterns. Physical inspection required before warranty extension.'
                                            : 'Battery health is optimal. You can extend your warranty for up to 3 years or 50,000 km.')
                                            : 'Please run the health analysis to check eligibility.'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </>
                );
        }
    };

    return (
        <div className={darkMode ? 'dark' : ''}>
            <div className="flex h-screen bg-[#F3F4F6] dark:bg-slate-900 font-sans text-slate-800 dark:text-white overflow-hidden transition-colors duration-300">

                {/* Sidebar */}
                <aside className={`${isSidebarOpen ? 'w-64' : 'w-20'} bg-white dark:bg-slate-800 h-full shadow-xl transition-all duration-300 flex flex-col z-20 border-r border-slate-100 dark:border-slate-700`}>
                    <div className="p-6 flex items-center justify-between">
                        <div className={`flex items-center space-x-2 ${!isSidebarOpen && 'hidden'}`}>
                            <div className="w-8 h-8 bg-teal-500 rounded-lg flex items-center justify-center">
                                <Zap className="text-white w-5 h-5" />
                            </div>
                            <span className="font-bold text-lg text-slate-800 dark:text-white">Batteryeze</span>
                        </div>
                        <button onClick={() => setSidebarOpen(!isSidebarOpen)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
                            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
                        </button>
                    </div>

                    <nav className="flex-1 mt-8 px-4 space-y-2">
                        <NavItem
                            icon={<LayoutDashboard />}
                            label="Dashboard"
                            active={currentView === 'dashboard'}
                            isOpen={isSidebarOpen}
                            onClick={() => setCurrentView('dashboard')}
                        />
                        <NavItem
                            icon={<Battery />}
                            label="Batteries"
                            active={currentView === 'batteries'}
                            isOpen={isSidebarOpen}
                            onClick={() => setCurrentView('batteries')}
                        />
                        <NavItem
                            icon={<ShieldCheck />}
                            label="Extend Warranty"
                            active={currentView === 'warranty'}
                            isOpen={isSidebarOpen}
                            onClick={() => setCurrentView('warranty')}
                        />
                        <NavItem
                            icon={<User />}
                            label="Profile & Settings"
                            active={currentView === 'profile'}
                            isOpen={isSidebarOpen}
                            onClick={() => setCurrentView('profile')}
                        />
                    </nav>
                </aside>

                {/* Main Content */}
                <main className="flex-1 overflow-y-auto relative" id="main-content">
                    <header className="bg-white dark:bg-slate-800 h-16 flex items-center justify-between px-8 border-b border-slate-100 dark:border-slate-700 sticky top-0 z-50">

                        <div>
                            <h2 className="text-xl font-bold text-slate-800 dark:text-white">
                                {selectedVehicle ? selectedVehicle.vehicle_id : "Select Vehicle"}
                            </h2>
                            <p className="text-xs text-slate-400">
                                Last updated on {new Date().toLocaleDateString()}
                            </p>
                        </div>

                        <div className="flex items-center gap-4">

                            {/* Theme Toggle */}
                            <button onClick={toggleTheme}
                                className="p-2 hover:bg-slate-700 rounded-full">
                                {darkMode ? <Sun size={20} /> : <Moon size={20} />}
                            </button>

                            {/* Profile */}
                            <div className="relative">

                                <button
                                    onClick={() => setShowDropdown(!showDropdown)}
                                    className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
                                    <User size={18} />
                                </button>

                                {showDropdown && (
                                    <div className="absolute right-0 mt-2 bg-slate-800 rounded-xl shadow-xl p-3 z-50">

                                        <p
                                            onClick={() => setCurrentView("profile")}
                                            className="cursor-pointer hover:text-teal-400">
                                            Profile
                                        </p>

                                        <p
                                            onClick={() => {
                                                localStorage.removeItem("userId");
                                                window.location.reload();
                                            }}
                                            className="cursor-pointer mt-2 hover:text-red-400">
                                            Logout
                                        </p>

                                    </div>
                                )}

                            </div>

                            {/* Export */}
                            <button
                                onClick={() => result && generatePDF('main-content', { ...formData, ...result })}
                                className="px-4 py-2 text-sm bg-teal-500 rounded-full">
                                Export Report
                            </button>

                        </div>

                    </header>
                    <div className="p-8 pb-32">
                        {renderContent()}
                    </div>

                    {/* Floating Chat */}
                    <ChatBot healthData={result} />

                </main>
            </div>
        </div >
    );
}

// Helpers
const NavItem = ({ icon, label, active, isOpen, onClick }) => (
    <button onClick={onClick} className={`flex items-center w-full p-3 rounded-xl transition-all ${active ? 'bg-teal-50 dark:bg-teal-900/30 text-teal-600 dark:text-teal-400 font-semibold shadow-sm' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'}`}>
        <span className="w-5 h-5">{icon}</span>
        {isOpen && <span className="ml-3 text-sm">{label}</span>}
    </button>
);

const MaterialStat = ({ label, value, unit, color }) => (
    <div>
        <p className={`text-lg font-bold ${color}`}>{typeof value === 'number' ? value.toLocaleString() : value} <span className="text-xs text-slate-400">{unit}</span></p>
        <p className="text-xs text-slate-400 font-medium uppercase mt-1">{label}</p>
    </div>
);

const DetailRow = ({ label, value, highlight }) => (
    <div className="flex justify-between items-center py-3 border-b border-gray-50 dark:border-slate-700 last:border-0">
        <span className="text-slate-500 dark:text-slate-400 text-sm">{label}</span>
        <span className={`font-medium ${highlight ? 'text-red-500' : 'text-slate-700 dark:text-slate-200'}`}>{value}</span>
    </div>
);

const InfoCard = ({ title, value, sub, icon }) => (
    <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 flex items-center space-x-4 transition-colors duration-200">
        <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded-full">{icon}</div>
        <div>
            <p className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase">{title}</p>
            <p className="text-xl font-bold text-slate-800 dark:text-white">{value}</p>
            <p className="text-xs text-slate-400 dark:text-slate-500">{sub}</p>
        </div>
    </div>
);

const PlanCard = ({ name, price, features, recommended, disabled, note }) => (
    <div className={`p-6 rounded-2xl border transition-all duration-200 ${recommended ? 'border-teal-500 ring-1 ring-teal-500 bg-teal-50/20 dark:bg-teal-900/20' : 'border-slate-100 dark:border-slate-700 bg-white dark:bg-slate-800'} ${disabled ? 'opacity-50 grayscale' : ''}`}>
        <div className="flex justify-between items-start mb-4">
            <h3 className="font-bold text-lg text-slate-800 dark:text-white">{name}</h3>
            {recommended && <span className="px-2 py-1 bg-teal-100 dark:bg-teal-900/50 text-teal-700 dark:text-teal-300 text-xs rounded-md font-semibold">Best Value</span>}
        </div>
        <div className="text-3xl font-bold text-slate-800 dark:text-white mb-2">{price}</div>
        {note && <p className="text-xs text-orange-500 font-medium mb-4">{note}</p>}
        <ul className="space-y-3 mb-6">
            {features.map((feat, i) => (
                <li key={i} className="flex items-center text-sm text-slate-600 dark:text-slate-300">
                    <CheckCircle className="w-4 h-4 text-teal-500 mr-2" />
                    {feat}
                </li>
            ))}
        </ul>
        <button disabled={disabled} className={`w-full py-2 rounded-lg font-medium text-sm transition-colors ${recommended ? 'bg-teal-500 text-white hover:bg-teal-600' : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'} ${disabled ? 'cursor-not-allowed' : ''}`}>
            {disabled ? 'Not Eligible' : 'Select Plan'}
        </button>
    </div>
);

export default App;
